//! # Uniform Binding Module
//!
//! This module defines the `UniformBinding` struct, which is responsible for binding uniform buffers
//! to the GPU rendering pipeline. It provides an abstraction that simplifies the process
//! of synchronizing uniform data (e.g., transformation matrices, camera properties) between
//! the CPU and GPU, using the `wgpu` library.
//!
//! ## Overview
//!
//! The `UniformBinding` struct encapsulates:
//!
//! - **Uniform buffer**: A buffer that holds data sent to GPU shaders, such as transformation matrices
//!   or other uniform data.
//! - **Bind group**: A mechanism that binds the uniform buffer to the GPU pipeline. This allows the
//!   buffer data to be made accessible within shaders.
//! - **Bind group layout**: A descriptor that describes how resources (e.g., uniform buffers) are
//!   accessed by the GPU pipeline. It defines the structure and visibility of bindings.
//!
//! By packaging these components together, the `UniformBinding` struct ensures that resources are
//! consistently managed and compatible with different stages of the GPU pipeline.
//!
//! ## Features
//!
//! - **GPU Resource Management**: Simplifies the process of creating and managing uniform GPU resources,
//!   adhering to `wgpu`'s strict resource model.
//! - **Efficient Buffer Updates**: Provides methods to update the uniform buffer with minimal
//!   performance overhead.
//! - **Shader Compatibility**: Ensures seamless integration with shaders through consistent binding
//!   and layout specifications.
//!
//! ## Example
//!
//! ```rust
//! use wgpu::Queue;
//!
//! // Create a uniform binding
//! let uniform_binding = UniformBinding::new(&device);
//!
//! // Update the uniform buffer with a new Model-View-Projection matrix
//! let mvp_matrix = nalgebra_glm::identity();
//! let uniform_buffer_data = UniformBuffer { mvp: mvp_matrix };
//! uniform_binding.update_buffer(&queue, 0, uniform_buffer_data);
//!
//! // Use `uniform_binding.bind_group` in your render pipeline
//! // e.g., setting it during a render pass
//! render_pass.set_bind_group(0, &uniform_binding.bind_group, &[]);
//! ```
//!
//! ## How It Works
//!
//! The `UniformBinding` struct is designed to simplify the process of managing uniform data that needs
//! to be transferred to GPU shaders. Here's an overview of its usage:
//!
//! 1. **Initialization**: A new `UniformBinding` instance is created using the `new` method, which:
//!    - Allocates a uniform buffer on the GPU using the `wgpu::Device`.
//!    - Defines a bind group layout to specify the binding structure.
//!    - Creates a bind group that links the uniform buffer to the pipeline.
//!
//! 2. **Updating Uniform Data**: The `update_buffer` method writes new data to the uniform buffer. This
//!    operation synchronizes the CPU-side changes with the GPU-side resources.
//!
//! 3. **Rendering**: Before rendering, the `bind_group` property is set in the render pipeline, allowing
//!    shaders to access the uniform buffer at a specified binding slot.
//!
//! ## Struct Components
//!
//! - `buffer`: The GPU buffer holding uniform data, created with usage flags for uniform and copy operations.
//! - `bind_group`: The resource that connects the buffer to the pipeline for GPU access.
//! - `bind_group_layout`: Describes how the bound resources are expected to behave in shaders.
//!
//! ## Note on Buffer Layout
//!
//! The data written to the uniform buffer must conform to the memory layout expected by the shader.
//! This is typically achieved by using `#[repr(C)]` struct annotations and deriving traits like
//! `bytemuck::Pod` and `bytemuck::Zeroable` for the uniform buffer's data type.

// Import the `UniformBuffer` struct, which represents the uniform data structure
// (e.g., transformation matrices) passed from the CPU to the GPU. 
// It is used to define the layout and contents of the uniform buffer
// managed by the `UniformBinding` struct.
use crate::uniform_buffer::UniformBuffer;

/// Represents the binding of a uniform buffer to the GPU pipeline.
///
/// This struct encapsulates all components required to bind a uniform buffer
/// to the rendering pipeline, including the GPU buffer itself, the bind group,
/// and the bind group layout. It efficiently manages the resources needed for
/// passing uniform data (e.g., transformation matrices) from the CPU to the GPU.
///
/// # Fields
///
/// - `buffer`: The GPU buffer that stores the uniform data. Typically updated
///   with transformation matrices or other uniform data using the `update_buffer` method.
/// - `bind_group`: Encapsulates the binding between the uniform buffer and the
///   rendering pipeline. It's responsible for making the `buffer` available in shaders.
/// - `bind_group_layout`: Layout specification describing how the uniform
///   buffer is exposed to the shaders. It defines the binding types and visibility.
///
/// # Example
///
/// Creating a new `UniformBinding` and updating the uniform buffer:
///
/// ```rust
/// // Create a uniform binding
/// let uniform_binding = UniformBinding::new(&device);
///
/// // Update the uniform buffer with a new MVP matrix
/// let mvp_matrix = nalgebra_glm::identity();
/// let uniform_buffer = UniformBuffer { mvp: mvp_matrix };
/// uniform_binding.update_buffer(&queue, 0, uniform_buffer);
///
/// // Use `uniform_binding.bind_group` in your render pipeline
/// ```
///
/// # GPU Resources
///
/// The `UniformBinding` struct ensures compatibility with `wgpu`'s resource model
/// by adhering to proper memory layouts and providing a clear API for updating GPU resources.
pub struct UniformBinding {
    /// The GPU buffer that stores the data for the uniform block.
    ///
    /// This buffer is used to transfer uniform data, such as transformation matrices,
    /// from the CPU to the GPU. It is created with usage flags `wgpu::BufferUsages::UNIFORM`
    /// and `wgpu::BufferUsages::COPY_DST`, allowing it to serve as a uniform buffer
    /// and be updated from the CPU.
    ///
    /// It's essential to maintain consistency in the memory layout of the data written
    /// to this buffer with the shader's expected uniform block layout. This is often ensured
    /// by using proper annotations (e.g., `#[repr(C)]`) and compatibility traits (like
    /// `bytemuck::Pod` and `bytemuck::Zeroable`) on the associated data structures.
    ///
    /// # Update Process
    ///
    /// This buffer is typically updated at runtime through the `update_buffer` method,
    /// which writes new values into it. Updates are optimized for minimizing
    /// performance overhead while synchronizing CPU and GPU states.
    pub buffer: wgpu::Buffer,

    /// The bind group that binds the uniform buffer to the GPU pipeline.
    ///
    /// This property establishes the connection between the uniform buffer and
    /// the shaders within the rendering pipeline. A `BindGroup` describes the actual
    /// GPU resource bindings and exposes them to the pipeline stages specified during its creation.
    ///
    /// - The `bind_group` is created using the `bind_group_layout`, ensuring that the bindings
    ///   are compatible with the layout expectations in the shader.
    /// - It binds the uniform buffer (`UniformBinding::buffer`) to a specific binding slot (e.g., binding 0),
    ///   making it available to the GPU for reading uniform data during rendering.
    ///
    /// The `bind_group` ensures that the same uniform buffer can be reused across different
    /// draw calls, with minimal setup overhead.
    pub bind_group: wgpu::BindGroup,

    /// The layout specification for the bind group.
    ///
    /// This property defines the structure of the bindings used by the `bind_group`,
    /// and it describes how the GPU expects the uniform buffer to be accessed
    /// in shaders. The `bind_group_layout` serves as a blueprint for creating
    /// compatible `bind_group`s and ensures the bindings are properly aligned
    /// with the shader's expectations.
    ///
    /// # Key Features
    ///
    /// - It establishes the visibility and types of resources in the uniform block.
    /// - Enforces a strict description of all bindings (e.g., buffers, textures) that
    ///   can be accessed by the GPU when executing shaders.
    /// - This specific layout is designed mainly to support uniform buffers for passing
    ///   data like transformation matrices (e.g., Model-View-Projection matrix).
    ///
    /// # Shader Compatibility
    ///
    /// The layout ensures that the uniform buffer is declared at binding `0` in the shader.
    /// For example, in WGSL:
    /// ```wgsl
    /// @group(0) @binding(0)
    /// var<uniform> uniform_data: Uniform;
    /// ```
    ///
    /// This alignment between the layout descriptor and the shader allows seamless
    /// integration between GPU resources and shader code.
    ///
    /// # Usage
    ///
    /// The `bind_group_layout` is used when constructing a bind group (`wgpu::BindGroup`)
    /// and when setting up a render pipeline, ensuring the pipeline knows how to manage
    /// and access the bound resources efficiently.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

/// Implementation of the `UniformBinding` struct.
///
/// This block provides the methods and utilities required to manage and update
/// GPU-bound resources encapsulated by the `UniformBinding` struct. These methods ensure
/// seamless interaction between the uniform buffer and the GPU pipeline, covering initialization,
/// data synchronization, and resource binding.
///
/// # Methods
///
/// - **`new`**: Creates and initializes a new `UniformBinding` instance, including the
///   uniform buffer, bind group layout, and bind group required for GPU resource binding.
/// - **`update_buffer`**: Updates the contents of the uniform buffer with new data at runtime,
///   ensuring efficient synchronization between CPU and GPU states.
///
/// The primary goal of these utility methods is to provide a developer-friendly
/// API for working with GPU resources while adhering to the strict requirements of the
/// `wgpu` resource model.
impl UniformBinding {
    /// Creates a new instance of `UniformBinding`.
    ///
    /// This function initializes a uniform buffer, a bind group layout, and a bind group,
    /// which are essential components for managing uniform data and binding them to the GPU pipeline.
    ///
    /// # Parameters
    ///
    /// - `device`: A reference to the `wgpu::Device` used for resource creation.
    ///   The device is responsible for managing GPU resources, and this parameter is necessary
    ///   for creating the uniform buffer, bind group layout, and bind group.
    ///
    /// # Returns
    ///
    /// A new `UniformBinding` instance containing:
    /// - A uniform buffer for holding uniform data.
    /// - A bind group layout that describes the structure of resource bindings.
    /// - A bind group that links the uniform buffer to the GPU pipeline.
    ///
    /// # Usage
    ///
    /// ```rust
    /// let uniform_binding = UniformBinding::new(&device);
    /// ```
    ///
    /// After this call, the `UniformBinding` instance can be used to pass uniform data
    /// to GPU shaders during rendering.
    ///
    /// # Notes
    ///
    /// - The uniform buffer is initialized with the default contents of `UniformBuffer`.
    /// - The bind group layout and bind group are configured to make the uniform buffer accessible
    ///   in the vertex shader stage at binding `0`.
    pub fn new(device: &wgpu::Device) -> Self {
        // A GPU buffer used for storing uniform data.
        //
        // This buffer is allocated on the GPU and serves as a storage location for uniform
        // data that is accessed by shaders during rendering. It is configured to support
        // usage as a **uniform buffer** and allows for dynamic updates as needed.
        //
        // # Key Features
        //
        // - The buffer is initialized with the default contents of the `UniformBuffer` struct.
        // - It is used to hold data such as transformation matrices required by the shaders.
        // - Supports two primary usage flags:
        //   - **`wgpu::BufferUsages::UNIFORM`**: Indicates the buffer is intended for use as a uniform buffer.
        //   - **`wgpu::BufferUsages::COPY_DST`**: Allows the buffer to be updated dynamically from the CPU.
        //
        // # GPU Integration
        //
        // This buffer plays a critical role in binding uniform data to the GPU pipeline, which
        // enables shaders to access the required uniform information during rendering.
        //
        // # Synchronization
        //
        // When updating the buffer's content during runtime using the `update_buffer` method,
        // care must be taken to ensure proper alignment and size matches the expectations of the GPU
        // and the associated shader code.
        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[UniformBuffer::default()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // A layout for the bind group that defines the structure of resource bindings.
        //
        // This layout describes how resources like buffers and textures are bound to the GPU pipeline.
        // It ensures that the GPU knows the type, stage visibility, and binding index for each resource.
        //
        // # Key Features
        //
        // - **Binding Index**: This layout has a single entry at binding `0`, representing the uniform buffer.
        // - **Shader Stage Visibility**: The uniform buffer is visible to the vertex shader stage only.
        // - **Binding Type**: The binding is configured as a `Uniform` buffer, which allows read-only
        //   access to the data during rendering.
        //
        // # Usage
        //
        // This layout is a blueprint for creating bind groups that link uniform data to specific
        // bindings in the shader. It supports efficient resource management and ensures compatibility with
        // the GPU pipeline.
        //
        // # Notes
        //
        // - The `has_dynamic_offset` field is set to `false`, which means that the data binding does not
        //   allow dynamic offsets.
        // - The `min_binding_size` field is set to `None` because the buffer size is fixed and defined
        //   by the structure of the `UniformBuffer`.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("uniform_bind_group_layout"),
        });

        // A bind group that links the uniform buffer to the GPU pipeline.
        //
        // This bind group represents a binding between the uniform buffer and its layout
        // in the GPU's resource bindings. It is used to pass uniform data from the CPU (via the `buffer`)
        // to the GPU during rendering.
        //
        // # Key Features
        //
        // - **Resource Binding**: Associates the uniform buffer with the binding index `0` specified
        //   in the bind group layout.
        // - **GPU Pipeline Integration**: Ensures that the uniform data becomes available to the GPU shaders
        //   during the rendering process.
        // - **Fixed Binding**: This bind group has a single entry for the uniform buffer, making it a simple
        //   and efficient way to manage the connection to the shader pipeline.
        //
        // # Usage
        //
        // The bind group is tied directly to the `bind_group_layout` and is required whenever
        // the shader requires access to the uniform buffer data. It simplifies the process of
        // providing uniform data to GPU resources.
        //
        // # Notes
        //
        // - The bind group must match the layout defined by `bind_group_layout` for compatibility.
        // - The `UniformBuffer` data structure backing the buffer is expected to align with the shader's
        //   expectations to avoid runtime errors.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        Self {
            buffer,
            bind_group,
            bind_group_layout,
        }
    }

    /// Updates the contents of the uniform buffer.
    ///
    /// This function allows dynamic updating of the uniform buffer's data at the specified
    /// offset. It ensures synchronization between the CPU state and GPU resources by writing
    /// new data into the buffer.
    ///
    /// # Parameters
    ///
    /// - `queue`: A reference to the `wgpu::Queue` used for queueing buffer updates.
    ///   The queue handles the submission of commands to the GPU.
    /// - `offset`: The offset (in bytes) at which the `uniform_buffer` data will be updated
    ///   in the GPU's uniform buffer.
    /// - `uniform_buffer`: The new data to be written into the uniform buffer.
    ///
    /// # Usage
    ///
    /// Call this method to update the uniform buffer during runtime. For example:
    ///
    /// ```rust
    /// uniform_binding.update_buffer(&queue, 0, UniformBuffer::default());
    /// ```
    ///
    /// This will write new data to the buffer, starting at the beginning of the allocated
    /// uniform buffer space.
    ///
    /// # Notes
    ///
    /// - Ensure that `offset` is aligned properly according to the GPU requirements for the
    ///   uniform buffer data.
    /// - The `uniform_buffer` provided must adhere to the expected layout of the buffer
    ///   on the GPU side.
    pub fn update_buffer(
        &mut self,
        queue: &wgpu::Queue,
        offset: wgpu::BufferAddress,
        uniform_buffer: UniformBuffer,
    ) {
        queue.write_buffer(
            &self.buffer,
            offset,
            bytemuck::cast_slice(&[uniform_buffer]),
        )
    }
}
