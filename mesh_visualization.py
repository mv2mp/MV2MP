from typing import Optional, Tuple

import jaxtyping as jt
import torch
import torch.nn as nn

if torch.cuda.is_available():
    from pytorch3d import (
        renderer as pytorch3d_renderer,
        structures as pytorch3d_structures,
    )


def to_homogeneous(arr: jt.Float[torch.Tensor, "* D"]) -> jt.Float[torch.Tensor, "* D+1"]:
    cat_part = torch.ones_like(arr[..., :1], dtype=arr.dtype)
    return torch.cat([arr, cat_part], dim=-1)


class MaskShader(nn.Module):
    def __init__(self):
        super().__init__()
        self.blend_params = pytorch3d_renderer.BlendParams(background_color=(0.0, 0.0, 0.0))

    def forward(self, fragments, meshes: 'pytorch3d_structures.Meshes', **kwargs) -> torch.Tensor:
        """
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        colors = meshes.sample_textures(fragments)
        blend_params = self.blend_params
        images = pytorch3d_renderer.hard_rgb_blend(colors, fragments, blend_params)
        return images


def prepare_pytorch3d_cams(
    intrinsics: jt.Float[torch.Tensor, "N 3 3"],
    extrinsics: jt.Float[torch.Tensor, "N 3 4"],
    img_size_hw: Tuple[int, int],
) -> 'pytorch3d_renderer.PerspectiveCameras':
    device = intrinsics.device
    assert extrinsics.device == device

    focal_lenghts = [(K[0, 0].item(), K[1, 1].item()) for K in intrinsics]
    principal_points = [(K[0, 2].item(), K[1, 2].item()) for K in intrinsics]

    cameras = pytorch3d_renderer.PerspectiveCameras(
        focal_length=focal_lenghts,
        principal_point=principal_points,
        image_size=[img_size_hw] * len(intrinsics),
        device=device,
        in_ndc=False,
    )

    return cameras


def render_mesh_with_custom_pt3d_shader(
    shader: 'pytorch3d_renderer.mesh.shader.ShaderBase',
    vertices: jt.Float[torch.Tensor, "V 3"],
    faces: jt.Int[torch.Tensor, "F 3"],
    intrinsics: jt.Float[torch.Tensor, "N 3 3"],
    extrinsics: jt.Float[torch.Tensor, "N 3 4"],
    img_size_hw: Tuple[int, int],
    vertices_post_rotate_axisflip: Tuple[int, int, int] = (-1, -1, 1),
    vertices_colors: Optional[jt.Float[torch.Tensor, "V 3"]] = None,
) -> jt.Float[torch.Tensor, "N H W 3"]:
    device = intrinsics.device
    assert device.type == "cuda", "device should be CUDA"
    assert vertices.device == device, "vertices should be on the same device as intrinsics"
    assert faces.device == device, "faces should be on the same device as intrinsics"
    assert extrinsics.device == device, "extrinsics should be on the same device as intrinsics"

    if vertices_colors is not None:
        assert vertices_colors.device == device, "vertices_colors should be on the same device as intrinsics"

    n_cams = len(intrinsics)

    cameras = prepare_pytorch3d_cams(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        img_size_hw=img_size_hw,
    )

    vertices = vertices.float()

    if vertices_colors is None:
        vertices_colors = torch.ones_like(vertices, device=device, dtype=torch.float32)

    vertices_in_cameras: jt.Float[torch.Tensor, "N V 3"] = torch.einsum(
        "nij,vj->nvi", extrinsics.float(), to_homogeneous(vertices)
    )
    vertices_in_cameras = vertices_in_cameras * torch.tensor(vertices_post_rotate_axisflip, device=device).float()

    raster_settings = pytorch3d_renderer.RasterizationSettings(
        image_size=img_size_hw,
        blur_radius=0.0,
        faces_per_pixel=16,
        bin_size=0,
    )

    renderer = pytorch3d_renderer.MeshRenderer(
        shader=shader,
        rasterizer=pytorch3d_renderer.MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    ).to(device)

    textures = pytorch3d_renderer.TexturesVertex([vertices_colors.float()] * n_cams)

    meshes = pytorch3d_structures.Meshes(
        [v for v in vertices_in_cameras],
        [faces] * n_cams,
        textures=textures,
    )

    return renderer.forward(meshes)[..., :3]


def render_mesh_as_mask(
    vertices: jt.Float[torch.Tensor, "V 3"],
    faces: jt.Int[torch.Tensor, "F 3"],
    intrinsics: jt.Float[torch.Tensor, "N 3 3"],
    extrinsics: jt.Float[torch.Tensor, "N 3 4"],
    img_size_hw: Tuple[int, int],
    vertices_post_rotate_axisflip: Tuple[int, int, int] = (-1, -1, 1),
    vertices_colors: Optional[jt.Float[torch.Tensor, "V 3"]] = None,
) -> jt.Float[torch.Tensor, "N H W 3"]:
    return render_mesh_with_custom_pt3d_shader(
        shader=MaskShader(),
        vertices=vertices,
        faces=faces,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        img_size_hw=img_size_hw,
        vertices_post_rotate_axisflip=vertices_post_rotate_axisflip,
        vertices_colors=vertices_colors,
    )


def render_mesh_with_lighting(
    vertices: jt.Float[torch.Tensor, "V 3"],
    faces: jt.Int[torch.Tensor, "F 3"],
    intrinsics: jt.Float[torch.Tensor, "N 3 3"],
    extrinsics: jt.Float[torch.Tensor, "N 3 4"],
    img_size_hw: Tuple[int, int],
    vertices_post_rotate_axisflip: Tuple[int, int, int] = (-1, -1, 1),
    vertices_colors: Optional[jt.Float[torch.Tensor, "V 3"]] = None,
    light_location: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> jt.Float[torch.Tensor, "N H W 3"]:
    cameras = prepare_pytorch3d_cams(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        img_size_hw=img_size_hw,
    )

    device = intrinsics.device
    assert vertices.device == device, "vertices should be on the same device as intrinsics"
    assert faces.device == device, "faces should be on the same device as intrinsics"
    assert extrinsics.device == device, "extrinsics should be on the same device as intrinsics"
    if vertices_colors is not None:
        assert vertices_colors.device == device, "vertices_colors should be on the same device as intrinsics"

    return render_mesh_with_custom_pt3d_shader(
        shader=pytorch3d_renderer.HardPhongShader(
            cameras=cameras,
            device=device,
            lights=pytorch3d_renderer.PointLights(device=device, location=[light_location]),
        ),
        vertices=vertices,
        faces=faces,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        img_size_hw=img_size_hw,
        vertices_post_rotate_axisflip=vertices_post_rotate_axisflip,
        vertices_colors=vertices_colors,
    )
