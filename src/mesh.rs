#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

use back::Backend as B;

extern crate gfx_hal as hal;
use hal::{PhysicalDevice, Device, Backend};

use ::gfx_helpers;

pub struct GpuBuffer {
    pub buffer : <B as Backend>::Buffer,
    pub memory : <B as Backend>::Memory,
	pub usage  : hal::buffer::Usage,
}

impl GpuBuffer {

	//TODO: Staging Buffer
	//TODO: memory property argument (CPU_VISIBLE, etc.) 
	pub fn new<T : Copy>(data : &[T], usage : hal::buffer::Usage, device : &back::Device, physical_device : &back::PhysicalDevice) -> GpuBuffer {
        
		let buffer_stride = std::mem::size_of::<T>() as u64;
        let buffer_len = data.len() as u64 * buffer_stride;
		
		let buffer_unbound = device.create_buffer(buffer_len, usage).unwrap();
        let buffer_req = device.get_buffer_requirements(&buffer_unbound);

		let upload_type = gfx_helpers::get_memory_type(physical_device, &buffer_req, hal::memory::Properties::CPU_VISIBLE);

        let buffer_memory = device.allocate_memory(upload_type, buffer_req.size).unwrap();
        let buffer = device.bind_buffer_memory(&buffer_memory, 0, buffer_unbound).unwrap();

        {
            let mut mapping_writer = device.acquire_mapping_writer::<T>(&buffer_memory, 0..buffer_req.size).unwrap();
            mapping_writer[0..data.len()].copy_from_slice(&data);
            device.release_mapping_writer(mapping_writer).unwrap();
        }

		GpuBuffer {
			buffer : buffer,
			memory : buffer_memory,
			usage  : usage,
		}
	}

	pub fn recreate<T : Copy>(&mut self, data : &[T], device : &back::Device, physical_device : &back::PhysicalDevice) {
		let new_buffer = GpuBuffer::new(data, self.usage, device, physical_device);
		self.buffer = new_buffer.buffer;
		self.memory = new_buffer.memory;
	}

    pub fn destroy(self, device: &back::Device) {
        device.destroy_buffer(self.buffer);
        device.free_memory(self.memory);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub a_pos: [f32; 3],
    pub a_col: [f32; 4],
    pub a_uv:  [f32; 2],
    pub a_norm: [f32; 3],
    pub a_joint_indices: [f32; 4],
    pub a_joint_weights: [f32; 4],
}

pub struct Mesh {
    pub vertex_buffer : GpuBuffer,
    pub index_buffer : GpuBuffer,
    pub index_count : u32,
}

impl Mesh {
    pub fn new(in_vertices : Vec<Vertex>, in_indices : Vec<u32>, device : &back::Device, physical_device : &back::PhysicalDevice ) -> Mesh {

        //Vertex Buffer Setup
		let vertex_buffer = GpuBuffer::new(&in_vertices, hal::buffer::Usage::VERTEX, device, physical_device);

        //Index Buffer Setup
		let index_buffer = GpuBuffer::new(&in_indices, hal::buffer::Usage::INDEX, device, physical_device);

        Mesh {
            vertex_buffer : vertex_buffer,
            index_buffer : index_buffer,
            index_count : in_indices.len() as u32,
        }
    }

    pub fn destroy(self, device: &back::Device) {
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}