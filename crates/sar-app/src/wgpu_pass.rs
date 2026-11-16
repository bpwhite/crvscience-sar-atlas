// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

//! Custom `egui_wgpu::CallbackTrait` hardware-instanced render pass (`wgpu 24` via `eframe 0.31`)
//! driving the 3-pass `chem_space.wgsl` pipeline at 120+ FPS for 50,000+ compounds.

use core::num::NonZeroU64;
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu, wgpu::util::DeviceExt as _};
use sar_core::{GpuCameraUniform, GpuEdgeInstance, GpuNodeInstance, HEATMAP_DIM};
use std::sync::Arc;

pub struct ChemSpaceGpuResources {
    pub heatmap_pipeline: wgpu::RenderPipeline,
    pub edge_pipeline: wgpu::RenderPipeline,
    pub node_pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub heatmap_texture: wgpu::Texture,
    pub node_buffer: wgpu::Buffer,
    pub node_capacity: usize,
    pub active_nodes: u32,
    pub edge_buffer: wgpu::Buffer,
    pub edge_capacity: usize,
    pub active_edges: u32,
}

impl ChemSpaceGpuResources {
    pub fn initialize(cc: &eframe::CreationContext<'_>) -> bool {
        let Some(wgpu_state) = cc.wgpu_render_state.as_ref() else {
            return false;
        };
        let device = &wgpu_state.device;
        let target_format = wgpu_state.target_format;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sar_atlas_chem_space_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/chem_space.wgsl").into()),
        });

        let uniform_size = std::mem::size_of::<GpuCameraUniform>() as u64;
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sar_camera_uniform"),
            contents: bytemuck::bytes_of(&GpuCameraUniform {
                pan: [0.0, 0.0],
                zoom: 1.0,
                aspect: 1.4,
                viewport_px: [900.0, 640.0],
                heatmap_alpha: 0.55,
                time_sec: 0.0,
                bounds_min: [-100.0, -100.0],
                bounds_span: [200.0, 200.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let heatmap_extent = wgpu::Extent3d {
            width: HEATMAP_DIM as u32,
            height: HEATMAP_DIM as u32,
            depth_or_array_layers: 1,
        };
        let heatmap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sar_chem_heatmap_tex"),
            size: heatmap_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let heatmap_view = heatmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let heatmap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sar_chem_heatmap_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sar_chem_space_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(uniform_size),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sar_chem_space_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&heatmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&heatmap_sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sar_chem_space_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let alpha_target = [Some(wgpu::ColorTargetState {
            format: target_format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let heatmap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sar_heatmap_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_heatmap"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_heatmap"),
                targets: &alpha_target,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let edge_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuEdgeInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sar_edge_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_edge"),
                buffers: &[edge_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_edge"),
                targets: &alpha_target,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let node_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuNodeInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let node_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sar_node_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_node"),
                buffers: &[node_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_node"),
                targets: &alpha_target,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let initial_cap = 4096usize;
        let node_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sar_node_instances"),
            size: (initial_cap * std::mem::size_of::<GpuNodeInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let edge_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sar_edge_instances"),
            size: (initial_cap * std::mem::size_of::<GpuEdgeInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(ChemSpaceGpuResources {
                heatmap_pipeline,
                edge_pipeline,
                node_pipeline,
                bind_group,
                uniform_buffer,
                heatmap_texture,
                node_buffer,
                node_capacity: initial_cap,
                active_nodes: 0,
                edge_buffer,
                edge_capacity: initial_cap,
                active_edges: 0,
            });
        true
    }
}

pub struct ChemSpacePaintCallback {
    pub camera: GpuCameraUniform,
    pub nodes: Arc<Vec<GpuNodeInstance>>,
    pub edges: Arc<Vec<GpuEdgeInstance>>,
    pub new_heatmap_rgba: Option<Arc<Vec<u8>>>,
}

impl egui_wgpu::CallbackTrait for ChemSpacePaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(res) = resources.get_mut::<ChemSpaceGpuResources>() else {
            return Vec::new();
        };

        queue.write_buffer(&res.uniform_buffer, 0, bytemuck::bytes_of(&self.camera));

        if let Some(rgba) = &self.new_heatmap_rgba {
            if rgba.len() == HEATMAP_DIM * HEATMAP_DIM * 4 {
                queue.write_texture(
                    res.heatmap_texture.as_image_copy(),
                    rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * (HEATMAP_DIM as u32)),
                        rows_per_image: Some(HEATMAP_DIM as u32),
                    },
                    wgpu::Extent3d {
                        width: HEATMAP_DIM as u32,
                        height: HEATMAP_DIM as u32,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        if self.edges.len() > res.edge_capacity {
            res.edge_capacity = self.edges.len().next_power_of_two().max(1024);
            res.edge_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sar_edge_instances_grown"),
                size: (res.edge_capacity * std::mem::size_of::<GpuEdgeInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        res.active_edges = self.edges.len() as u32;
        if !self.edges.is_empty() {
            queue.write_buffer(&res.edge_buffer, 0, bytemuck::cast_slice(&self.edges));
        }

        if self.nodes.len() > res.node_capacity {
            res.node_capacity = self.nodes.len().next_power_of_two().max(1024);
            res.node_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sar_node_instances_grown"),
                size: (res.node_capacity * std::mem::size_of::<GpuNodeInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        res.active_nodes = self.nodes.len() as u32;
        if !self.nodes.is_empty() {
            queue.write_buffer(&res.node_buffer, 0, bytemuck::cast_slice(&self.nodes));
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<ChemSpaceGpuResources>() else {
            return;
        };

        render_pass.set_bind_group(0, &res.bind_group, &[]);

        // Pass 1: Continuous 256x256 bilinear field + iso-contours
        render_pass.set_pipeline(&res.heatmap_pipeline);
        render_pass.draw(0..6, 0..1);

        // Pass 2: Instanced SALI & Tanimoto similarity edges
        if res.active_edges > 0 {
            render_pass.set_pipeline(&res.edge_pipeline);
            render_pass.set_vertex_buffer(0, res.edge_buffer.slice(..));
            render_pass.draw(0..6, 0..res.active_edges);
        }

        // Pass 3: Instanced SDF compound nodes + alert/Pareto halos
        if res.active_nodes > 0 {
            render_pass.set_pipeline(&res.node_pipeline);
            render_pass.set_vertex_buffer(0, res.node_buffer.slice(..));
            render_pass.draw(0..6, 0..res.active_nodes);
        }
    }
}
