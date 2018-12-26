use ::hal;
use ::back;
use ::B;
use ::gfx_helpers;

use hal::{Device, Backend};

pub struct GpuBuffer {
    pub buffer : <B as Backend>::Buffer,
    pub memory : <B as Backend>::Memory,
	pub usage  : hal::buffer::Usage,
	pub memory_properties : hal::memory::Properties,
	pub count  : u32,
}

impl GpuBuffer {

	pub fn new<T : Copy>(	data : &[T], 
						 	usage : hal::buffer::Usage, 
							memory_properties: hal::memory::Properties, 
							device : &back::Device, 
							physical_device : &back::PhysicalDevice) 
	-> GpuBuffer {
        
		let use_staging_buffer = (memory_properties & hal::memory::Properties::CPU_VISIBLE) != hal::memory::Properties::CPU_VISIBLE;
		let upload_usage = if use_staging_buffer { hal::buffer::Usage::TRANSFER_SRC } else { usage };
		let upload_memory_properties = if use_staging_buffer { hal::memory::Properties::CPU_VISIBLE } else { memory_properties };

		let buffer_stride = std::mem::size_of::<T>() as u64;
        let buffer_len = data.len() as u64 * buffer_stride;
		
		let upload_buffer_unbound = device.create_buffer(buffer_len, upload_usage).unwrap();
        let upload_buffer_req = device.get_buffer_requirements(&upload_buffer_unbound);

		let upload_type = gfx_helpers::get_memory_type(physical_device, &upload_buffer_req, upload_memory_properties);

        let upload_buffer_memory = device.allocate_memory(upload_type, upload_buffer_req.size).unwrap();
        let upload_buffer = device.bind_buffer_memory(&upload_buffer_memory, 0, upload_buffer_unbound).unwrap();

        {
            let mut mapping_writer = device.acquire_mapping_writer::<T>(&upload_buffer_memory, 0..upload_buffer_req.size).unwrap();
            mapping_writer[0..data.len()].copy_from_slice(&data);
            device.release_mapping_writer(mapping_writer).unwrap();
        }

		if use_staging_buffer {
			//TODO: copy upload_buffer to final buffer using Device::copy_buffer
		}

		GpuBuffer {
			buffer 			  : upload_buffer,
			memory 			  : upload_buffer_memory,
			usage  			  : usage,
			memory_properties : memory_properties, 
			count 			  : data.len() as u32,
		}
	}

	fn recreate<T : Copy>(&mut self, data : &[T], device : &back::Device, physical_device : &back::PhysicalDevice) {
		let new_buffer = GpuBuffer::new(data, self.usage, self.memory_properties, device, physical_device);
		self.buffer = new_buffer.buffer;
		self.memory = new_buffer.memory;
		self.count  = data.len() as u32;
	}

	pub fn reupload<T : Copy>(&mut self, data: &[T], device : &back::Device, physical_device : &back::PhysicalDevice) {
		if data.len() as u32 > self.count {
			self.recreate(data, device, physical_device);
		} else {
			let mut mapping_writer = device.acquire_mapping_writer::<T>(&self.memory, 0..(self.count as u64 * (std::mem::size_of::<T>() as u64))).unwrap();
			mapping_writer[0..data.len()].copy_from_slice(&data);
			device.release_mapping_writer(mapping_writer).unwrap();
		}
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
    pub index_buffer : Option<GpuBuffer>,
}

impl Mesh {
    pub fn new(in_vertices : Vec<Vertex>, in_indices : Option<Vec<u32>>, device : &back::Device, physical_device : &back::PhysicalDevice ) -> Mesh {
        Mesh {
			//TODO: change these to Device Local when staging buffer is implemented
            vertex_buffer : GpuBuffer::new(&in_vertices, hal::buffer::Usage::VERTEX, hal::memory::Properties::CPU_VISIBLE, device, physical_device),
            index_buffer  : in_indices.map(|in_indices| GpuBuffer::new(&in_indices, hal::buffer::Usage::INDEX, hal::memory::Properties::CPU_VISIBLE, device, physical_device)),
        }
    }

	pub fn record_draw_commands<Level : hal::command::Level>( &self, encoder : &mut hal::command::RenderPassInlineEncoder<B, Level>, instance_count : u32)
	{
		encoder.bind_vertex_buffers(0, Some((&self.vertex_buffer.buffer, 0)));

		match &self.index_buffer {
			Some(index_buffer) => {
				encoder.bind_index_buffer(hal::buffer::IndexBufferView {
					buffer: &index_buffer.buffer,
					offset: 0,
					index_type: hal::IndexType::U32,
				});
				encoder.draw_indexed(0..index_buffer.count, 0, 0..instance_count);
			},
			None => {
				encoder.draw(0..self.vertex_buffer.count, 0..instance_count);
			}
		}
	}

    pub fn destroy(self, device: &back::Device) {
        self.vertex_buffer.destroy(device);
		match self.index_buffer {
			Some(gpu_buffer) => gpu_buffer.destroy(device),
			None => {},
		}
    }
}