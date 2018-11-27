#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

extern crate gfx_hal as hal;
use hal::PhysicalDevice;

//TODO: handle multiple desired memory properties
pub fn get_memory_type( physical_device: &back::PhysicalDevice, 
						memory_requirements : &hal::memory::Requirements, 
						desired_memory_property : hal::memory::Properties) 
-> hal::MemoryTypeId {
	physical_device.memory_properties().memory_types
		.iter()
		.enumerate()
		.position(|(id, mem_type)| {
			memory_requirements.type_mask & (1 << id) != 0 &&
			mem_type.properties.contains(desired_memory_property)
		})
		.unwrap()
		.into()
}