#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

use back::Backend as B;

extern crate gfx_hal as hal;
use hal::{PhysicalDevice, Device, Backend};

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub a_pos: [f32; 3],
    pub a_col: [f32; 4],
    pub a_uv:  [f32; 2],
    pub a_norm: [f32; 3],
    pub a_joint_indices: [f32; 4],
    pub a_joint_weights: [f32; 4],
}

pub struct GpuBuffer {
    pub buffer : <B as Backend>::Buffer,
    pub memory : <B as Backend>::Memory,
}

impl GpuBuffer {
    fn destroy(self, device: &back::Device) {
        device.destroy_buffer(self.buffer);
        device.free_memory(self.memory);
    }
}

pub struct Mesh {
    pub vertex_buffer : GpuBuffer,
    pub index_buffer : GpuBuffer,
    pub index_count : u32,
}

impl Mesh {
    pub fn new(in_vertices : Vec<Vertex>, in_indices : Vec<u32>, device : &back::Device, physical_device : &back::PhysicalDevice ) -> Mesh {

        let memory_types = physical_device.memory_properties().memory_types;

        //TODO: Staging Buffers

        //Vertex Buffer Setup
        let buffer_stride = std::mem::size_of::<Vertex>() as u64;
        let buffer_len = in_vertices.len() as u64 * buffer_stride;
        let buffer_unbound = device.create_buffer(buffer_len, hal::buffer::Usage::VERTEX).unwrap();
        let buffer_req = device.get_buffer_requirements(&buffer_unbound);

        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
            }).unwrap().into();

        let vertex_buffer_memory = device.allocate_memory(upload_type, buffer_req.size).unwrap();
        let vertex_buffer = device.bind_buffer_memory(&vertex_buffer_memory, 0, buffer_unbound).unwrap();

        {
            let mut vertices = device.acquire_mapping_writer::<Vertex>(&vertex_buffer_memory, 0..buffer_req.size).unwrap();
            vertices[0..in_vertices.len()].copy_from_slice(&in_vertices);
            device.release_mapping_writer(vertices).unwrap();
        }

        //Index Buffer Setup
        let index_buffer_stride = std::mem::size_of::<u32>() as u64;
        let index_buffer_len = in_indices.len() as u64 * index_buffer_stride;
        let index_buffer_unbound = device.create_buffer(index_buffer_len, hal::buffer::Usage::INDEX).unwrap();
        let index_buffer_req = device.get_buffer_requirements(&index_buffer_unbound);

        let index_upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                index_buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
            }).unwrap().into();

        let index_buffer_memory : <B as Backend>::Memory = device.allocate_memory(index_upload_type, index_buffer_req.size).unwrap();
        let index_buffer : <B as Backend>::Buffer = device.bind_buffer_memory(&index_buffer_memory, 0, index_buffer_unbound).unwrap();
        {
            let mut indices = device.acquire_mapping_writer::<u32>(&index_buffer_memory, 0..index_buffer_req.size).unwrap();
            indices[0..in_indices.len()].copy_from_slice(&in_indices);
            device.release_mapping_writer(indices).unwrap();
        }

        Mesh {
            vertex_buffer : GpuBuffer {
                buffer : vertex_buffer,
                memory : vertex_buffer_memory,
            },
            index_buffer : GpuBuffer {
                buffer : index_buffer,
                memory : index_buffer_memory,
            },
            index_count : in_indices.len() as u32,
        }
    }

    pub fn destroy(self, device: &back::Device) {
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}