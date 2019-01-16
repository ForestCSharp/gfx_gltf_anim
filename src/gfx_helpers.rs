use ::back;
use ::hal;
use hal::PhysicalDevice;

//TODO: handle multiple desired memory properties
pub fn get_memory_type( physical_device: &back::PhysicalDevice,
						memory_requirements : &hal::memory::Requirements, 
						desired_memory_properties : hal::memory::Properties) 
-> hal::MemoryTypeId {
	physical_device.memory_properties().memory_types
		.iter()
		.enumerate()
		.position(|(id, mem_type)| {
			memory_requirements.type_mask & (1 << id) != 0 &&
			mem_type.properties.contains(desired_memory_properties)
		})
		.unwrap()
		.into()
}

pub struct DeviceState {
	pub device : back::Device,
	pub physical_device : back::PhysicalDevice,
}