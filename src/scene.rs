//! # Scene
//!
//! The `scene` module provides structures and implementations for managing and rendering
//! a 3D scene. It focuses on creating and handling the resources required to render a 3D
//! object using the GPU, such as vertex buffers, index buffers, uniform buffers, and the
//! render pipeline.
//!
//! ## Overview
//!
//! This module is built around the `Scene` struct, which represents a 3D drawable object or
//! environment. It contains the necessary data and functionality to render objects onto the
//! screen. This includes:
//!
//! - Vertex and index buffers for object geometry data.
//! - Transformation matrices to manipulate object position, rotation, and scale.
//! - A uniform buffer that passes model-view-projection (MVP) matrices to shaders.
//! - A rendering pipeline defining graphics state and the connection between shaders, buffers, and the GPU.
//!
//! This implementation leverages `wgpu` for GPU management and rendering, and `nalgebra-glm` for matrix math.
//!
//! ## Key Features
//! - Create 3D scenes with vertex and index data through the `Scene` struct.
//! - Manage uniform buffers for frame-dependent updates like transformations.
//! - Render the scene through efficient GPU pipelines and commands.
//! - Update transformations dynamically based on frame data, time, or user input.
//!
//! ## Example Usage
//! ```rust
//! // 1. Create a new scene instance.
//! let scene = Scene::new(&device, surface_format);
//!
//! // 2. Update the scene with the current frame parameters.
//! scene.update(&queue, aspect_ratio, delta_time);
//!
//! // 3. Render the scene.
//! let mut render_pass = encoder.begin_render_pass(&render_pass_descriptor);
//! scene.render(&mut render_pass);
//! ```
//!
//! ## Dependencies
//!
//! This module uses the following dependencies:
//! - **`wgpu`** for GPU interfaces and rendering.
//! - **`nalgebra-glm`** for mathematical operations, especially matrix manipulations.
//!
//! These libraries are essential for building a performant and flexible 3D rendering pipeline.
//!
//! ## Structure
//!
//! The `Scene` struct has clearly defined fields and methods aimed at creating and managing
//! components needed for real-time rendering. Below is a summary of its key fields and functions:
//!
//! ### Fields
//! - **`model (Mat4)`**: Stores the transformation (local-to-world) applied to the 3D object.
//! - **`vertex_buffer (wgpu::Buffer)`**: Holds vertex geometry data (positions, colors, etc.).
//! - **`index_buffer (wgpu::Buffer)`**: Stores indices for efficient vertex reuse.
//! - **`uniform (UniformBinding)`**: Manages the uniform buffer for shader parameters.
//! - **`pipeline (wgpu::RenderPipeline)`**: Specifies how the GPU renders using shaders and other settings.
//!
//! ### Methods
//! - **`new()`**: Creates and initializes a new `Scene` instance, including buffers and pipeline.
//! - **`update()`**: Adjusts the scene's state, such as the transformation matrix, to reflect changes in time or user input.
//! - **`render()`**: Issues draw commands to render the `Scene` using the initialized GPU state.
//!
//! ## Design Goals
//! The `scene` module is designed to abstract away complex GPU operations and provide developers
//! with an easy-to-use API for rendering objects in 3D. While providing flexibility for customizations,
//! it also ensures performance is optimized when interacting with the GPU.

// Importing the `Renderer` struct from the `renderer` module, which interacts with this module 
// to render the 3D scene by utilizing the GPU resources and rendering pipelines provided 
// by the `Renderer`. This integration enables seamless rendering of the scene alongside GUI elements.
use crate::renderer::Renderer;

// Importing the `Vertex` struct and the `VERTICES` array from the `vertex` module.
// - `Vertex` represents the structure for a single vertex, typically containing position, 
//   normal, color, or texture coordinate data. It defines the layout of our vertex data.
// - `VERTICES` is an array that contains the actual vertex data used to construct
//   the geometry of the 3D objects in the scene. This data is uploaded to the GPU
//   and utilized in the rendering process.
use crate::vertex::{Vertex, VERTICES};

// Importing the `UniformBuffer` struct, which represents the uniform buffer used 
// to pass data such as the Model-View-Projection (MVP) matrix from the CPU to the GPU.
// It is used in the `Scene` module to manage per-frame transformation data for rendering.
use crate::uniform_buffer::UniformBuffer;

// Importing the `UniformBinding` struct, which handles the uniform buffer and its bindings.
// It is used in the `Scene` module to manage GPU resources for passing data like transformation
// matrices (e.g., Model-View-Projection matrix) to shaders.
use crate::uniform_binding::UniformBinding;

// Importing the `INDICES` array from the crate root, which defines the index order for vertex rendering.
// This array is used to create the index buffer in the `Scene` struct, enabling efficient reuse of vertex data
// and defining the triangles to be drawn by referencing the vertices in the correct order.
use crate::INDICES;

// Importing the `SHADER_SOURCE` constant, which contains the WGSL shader code used to define
// the vertex and fragment stages for rendering. This shader is used in the render pipeline
// to transform vertices and determine the color of rendered fragments.
use crate::SHADER_SOURCE;

/// Represents a 3D scene that contains a model, its associated buffers, and the
/// rendering pipeline configuration.
///
/// The `Scene` struct is responsible for managing the necessary resources for
/// rendering a 3D object to the screen, such as vertex and index buffers,
/// transformation matrices, as well as a uniform buffer for shader parameters.
///
/// # Fields
///
/// - `model`: The model transformation matrix (4x4) used to apply transformations
///   like translation, rotation, and scaling to the scene's object.
/// - `vertex_buffer`: A `wgpu::Buffer` that contains the vertex data for the object.
/// - `index_buffer`: A `wgpu::Buffer` used to store the index data defining
///   the object's geometry.
/// - `uniform`: A `UniformBinding` that manages the uniform buffer for shaders.
///   This typically includes the model-view-projection (MVP) matrix.
/// - `pipeline`: A `wgpu::RenderPipeline` that defines how the scene is rendered.
///
/// # Methods
///
/// - `new`: Creates a new `Scene` instance with a given `wgpu::Device` and
///   `wgpu::TextureFormat`. This method initializes all necessary resources.
/// - `render`: Encodes the commands to render the scene within a `wgpu::RenderPass`.
/// - `update`: Updates the scene's state, such as the transformation matrix, based on
///   input parameters like the aspect ratio and elapsed time.
///
/// # Example
///
/// ```rust
/// // Create a new scene with the device and surface format.
/// let scene = Scene::new(&device, surface_format);
///
/// // Update the scene before rendering.
/// scene.update(&queue, aspect_ratio, delta_time);
///
/// // Render the scene.
/// let mut render_pass = encoder.begin_render_pass(&render_pass_descriptor);
/// scene.render(&mut render_pass);
/// ```
pub struct Scene {
    /// The model transformation matrix (4x4) used for applying transformations
    /// such as translation, rotation, and scaling to the objects in the scene.
    /// This matrix defines the local-to-world space transformation of the
    /// rendered object and impacts how the object is positioned, oriented, and
    /// scaled within the 3D world. It is typically updated in the `update` method
    /// of the `Scene` based on the current frame's parameters (e.g., elapsed time).
    pub model: nalgebra_glm::Mat4,

    /// A `wgpu::Buffer` that stores the vertex data for the objects in the scene.
    ///
    /// This buffer contains the positions, normals, texture coordinates, or other
    /// attributes of the vertices that define the geometry of the object(s) being rendered.
    /// It is used by the GPU during the rendering process to accurately
    /// display the 3D object. The data in this buffer is set during the creation
    /// of the `Scene` instance and remains immutable during runtime.
    pub vertex_buffer: wgpu::Buffer,

    /// A `wgpu::Buffer` used to store the index data for defining the geometry of the object(s) in the scene.
    ///
    /// The index buffer contains indices pointing to entries in the vertex buffer, enabling the GPU to render
    /// complex shapes efficiently by reusing vertices. This technique is called indexed drawing and minimizes
    /// the amount of vertex data stored and transferred to the GPU. The data in this buffer is immutable during
    /// runtime and is set during the creation of the `Scene` instance.
    pub index_buffer: wgpu::Buffer,

    /// A `UniformBinding` that manages the uniform buffer for shaders.
    ///
    /// This uniform is primarily used to pass data, such as the model-view-projection (MVP)
    /// matrix, to the shaders during rendering. The transformation matrices are updated
    /// each frame based on the scene's state, enabling dynamic rendering of objects.
    ///
    /// The `update` method of the `Scene` updates the uniform buffer with the latest data
    /// before rendering. This ensures that the transformations, such as rotation or scaling,
    /// are reflected appropriately in the rendered scene.
    pub uniform: UniformBinding,

    /// The `wgpu::RenderPipeline` used to define how the scene is rendered.
    ///
    /// This pipeline encapsulates the GPU state and specifies the shader programs,
    /// rasterizer settings, blending modes, and other configurations required
    /// during the rendering process. The pipeline ensures that the GPU renders
    /// the scene consistently according to the specified graphics pipeline state.
    ///
    /// It is created during the initialization of the `Scene` via the `create_pipeline`
    /// method, which sets up the vertex and fragment shaders, as well as defines how
    /// the vertex data and output color formats are processed.
    pub pipeline: wgpu::RenderPipeline,
}

/// Implementation of methods for the `Scene` struct.
///
/// The `Scene` struct represents a drawable 3D object or environment that maintains
/// the resources required for rendering. This implementation provides methods for
/// initializing, updating, and rendering the scene.
///
/// # Methods
///
/// - `new`: Creates a new `Scene` with the given `wgpu::Device` and `wgpu::TextureFormat`.
///   This initializes the buffers, shaders, and rendering pipeline necessary to draw
///   the contents of the scene.
/// - `render`: Encodes commands for rendering the scene using the provided `wgpu::RenderPass`.
///   It binds the vertex, index buffers, and the appropriate pipeline to execute the draw commands.
/// - `update`: Updates the scene's transformation matrices and associated uniform buffer
///   based on the current frame state. This includes operations like updating the
///   model-view-projection matrix to account for elapsed time or the aspect ratio.
///
/// This implementation relies on external tools and libraries such as `nalgebra-glm`
/// for matrix math and `wgpu` for interfacing with the GPU.
impl Scene {
    /// Creates a new `Scene` instance with the necessary GPU resources for rendering.
    ///
    /// This method sets up the vertex buffer, index buffer, uniform buffer, and
    /// render pipeline needed to render a 3D scene. It initializes these resources
    /// using the provided `wgpu::Device` and `wgpu::TextureFormat`.
    ///
    /// # Parameters
    ///
    /// - `device`: A reference to the `wgpu::Device`, which is used to create and manage GPU resources.
    /// - `surface_format`: The `wgpu::TextureFormat` that defines the texture format for the rendering target.
    ///
    /// # Returns
    ///
    /// A new `Scene` instance containing all the initialized rendering resources.
    ///
    /// # Panics
    ///
    /// This method may panic if resource creation fails, such as when there is a problem allocating GPU memory
    /// or if the provided vertex/index data is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// let scene = Scene::new(&device, surface_format);
    /// ```
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        // The `wgpu::Buffer` that stores vertex data for the `Scene`.
        //
        // This buffer contains the vertices required to define the shape or geometry
        // of the 3D object(s) in the scene. The data includes attributes like position,
        // color, normals, or texture coordinates, depending on the defined vertex structure.
        //
        // The vertex buffer is uploaded to GPU memory and is used during the rendering process
        // in the vertex shader stage.
        //
        // It is initialized using `wgpu::util::DeviceExt::create_buffer_init` for simplicity,
        // and the contents are sourced from the `VERTICES` array.
        let vertex_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );

        // The `wgpu::Buffer` that stores index data for the `Scene`.
        //
        // This buffer contains the indices that define how the vertices in the vertex buffer
        // are connected to form geometric primitives, such as triangles, lines, or points.
        //
        // Using an index buffer allows for efficient reuse of vertex data by referencing
        // shared vertices rather than duplicating them for each primitive.
        //
        // The index buffer is allocated in GPU memory and is used during the rendering process
        // in conjunction with the vertex buffer. It is especially useful when rendering
        // complex models or scenes with a high degree of shared geometry.
        //
        // It is initialized using `wgpu::util::DeviceExt::create_buffer_init` for ease of
        // creation, and its contents are sourced from the `INDICES` array.
        let index_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("index Buffer"),
                contents: bytemuck::cast_slice(&INDICES),
                usage: wgpu::BufferUsages::INDEX,
            },
        );

        // The `UniformBinding` structure that handles the uniform buffer and its associated bind group.
        //
        // This component is responsible for storing and managing the uniform data required for the
        // `Scene`, such as the model-view-projection (MVP) matrix. The MVP matrix is used to
        // transform 3D geometry from model space to screen space.
        //
        // The `UniformBinding` includes a buffer that resides on the GPU, where it is updated
        // with transformed data via the `update` method. Additionally, it holds a bind group that
        // is used to bind the uniform data to the rendering pipeline during the draw call.
        //
        // The uniform buffer enables dynamic, per-frame updates of transformation data, ensuring
        // that rendered objects are always correctly positioned and oriented on the screen.
        //
        // It is initialized using the `UniformBinding::new` method, which sets up the buffer and
        // bind group using the provided `wgpu::Device`.
        let uniform = UniformBinding::new(device);

        // The `RenderPipeline` used to render the `Scene`.
        //
        // This pipeline encapsulates the entire GPU state needed for rendering, including the
        // shader programs, fixed-function state, and output formats. It is responsible for defining
        // how the vertex and fragment shaders are executed and how the results are written to the
        // render targets.
        //
        // The pipeline is created using the `Self::create_pipeline` method, which specifies
        // details such as the vertex and fragment shader modules, input vertex layout, and
        // render target configurations.
        //
        // The `RenderPipeline` is a core component of the rendering process, binding together
        // the rendering state and ensuring that the `Scene` is drawn correctly.
        let pipeline = Self::create_pipeline(device, surface_format, &uniform);

        Self {
            model: nalgebra_glm::Mat4::identity(),
            uniform,
            pipeline,
            vertex_buffer,
            index_buffer,
        }
    }

    /// Encodes and submits rendering commands for the `Scene` to the given `wgpu::RenderPass`.
    ///
    /// This method takes a mutable reference to a `wgpu::RenderPass` and binds the vertex
    /// buffer, index buffer, uniform bind group, and the render pipeline. It then issues
    /// the necessary commands to draw the object(s) represented by the `Scene`.
    ///
    /// # Parameters
    ///
    /// - `renderpass`: A mutable reference to the `wgpu::RenderPass` where the drawing will occur.
    ///
    /// # How it works
    ///
    /// 1. Configures the render pass with the render pipeline stored in this `Scene`.
    /// 2. Binds the uniform bind group at the appropriate binding point (set 0).
    /// 3. Sets up the vertex and index buffers for the GPU.
    /// 4. Issues the draw command using the index buffer.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Assuming `scene` is an instance of `Scene` and `render_pass` is a valid render pass.
    /// scene.render(&mut render_pass);
    /// ```
    pub fn render<'rpass>(&'rpass self, renderpass: &mut wgpu::RenderPass<'rpass>) {
        renderpass.set_pipeline(&self.pipeline);
        renderpass.set_bind_group(0, &self.uniform.bind_group, &[]);

        renderpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        renderpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        renderpass.draw_indexed(0..(INDICES.len() as _), 0, 0..1);
    }

    /// Updates the transformation and uniform data of the `Scene`.
    ///
    /// This method recalculates the model-view-projection (MVP) matrix
    /// and updates the uniform buffer with the new transformation data.
    ///
    /// # Parameters
    ///
    /// - `queue`: A reference to the `wgpu::Queue` used to upload updated uniform data to the GPU.
    /// - `aspect_ratio`: The aspect ratio of the rendering surface (width / height).
    /// - `delta_time`: The time elapsed since the last update, in seconds. Used for animated transformations.
    ///
    /// # How It Works
    ///
    /// 1. Calculates a perspective projection matrix based on the specified `aspect_ratio` and a fixed field of view.
    /// 2. Creates a view matrix for a fixed camera position and look-at target.
    /// 3. Updates the model matrix by applying a rotation around the Y-axis. The speed of the rotation is scaled by `delta_time`.
    /// 4. Combines the projection, view, and model matrices to create the MVP matrix.
    /// 5. Updates the uniform buffer with the newly-calculated MVP matrix using the provided `queue`.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Assuming `scene` is an instance of `Scene`, `queue` is a valid wgpu::Queue,
    /// // `aspect_ratio` is a float, and `delta_time` has been calculated.
    /// scene.update(&queue, aspect_ratio, delta_time);
    /// ```
    pub fn update(&mut self, queue: &wgpu::Queue, aspect_ratio: f32, delta_time: f32) {
        // A perspective projection matrix.
        //
        // This matrix converts 3D coordinates into 2D clip space coordinates
        // by applying a perspective transformation. It is calculated using the
        // aspect ratio of the rendering surface, a fixed field of view (in radians),
        // and near and far clipping planes.
        //
        // - `aspect_ratio`: The ratio of the rendering surface's width to its height.
        // - `80_f32.to_radians()`: The field of view (FOV) in radians, representing the vertical angle of the camera's view.
        // - `0.1`: The near clipping plane, representing the minimum distance from the camera where objects are visible.
        // - `1000.0`: The far clipping plane, representing the maximum distance from the camera where objects are visible.
        let projection =
            nalgebra_glm::perspective_lh_zo(aspect_ratio, 80_f32.to_radians(), 0.1, 1000.0);

        // A view matrix for the camera.
        //
        // The view matrix transforms world coordinates into the camera's coordinate space.
        // It is calculated based on the camera's position, the target point it is looking at,
        // and an up direction vector.
        //
        // - `&nalgebra_glm::vec3(0.0, 0.0, 3.0)`: The position of the camera in world space.
        // - `&nalgebra_glm::vec3(0.0, 0.0, 0.0)`: The target point that the camera is looking at.
        // - `&nalgebra_glm::Vec3::y()`: The up direction vector, which aligns the camera's orientation.
        let view = nalgebra_glm::look_at_lh(
            &nalgebra_glm::vec3(0.0, 0.0, 3.0),
            &nalgebra_glm::vec3(0.0, 0.0, 0.0),
            &nalgebra_glm::Vec3::y(),
        );

        self.model = nalgebra_glm::rotate(
            &self.model,
            30_f32.to_radians() * delta_time,
            &nalgebra_glm::Vec3::y(),
        );
        self.uniform.update_buffer(
            queue,
            0,
            UniformBuffer {
                mvp: projection * view * self.model,
            },
        );
    }

    /// Creates a render pipeline for the `Scene`.
    ///
    /// This function sets up a graphics pipeline that specifies how vertices and
    /// fragments will be processed and rendered to the screen. It takes configuration
    /// parameters like the surface format and uniform bind group layout needed for
    /// the pipeline's creation.
    ///
    /// # Parameters
    ///
    /// - `device`: A reference to the `wgpu::Device` used to create GPU resources.
    /// - `surface_format`: The `wgpu::TextureFormat` of the rendering surface, which
    ///   determines the format of the frame buffer to render into.
    /// - `uniform`: A reference to the `UniformBinding` object, which provides the
    ///   bind group layout used to bind the uniform buffer for shaders.
    ///
    /// # How it Works
    ///
    /// 1. Compiles the shaders using the provided WGSL shader source.
    /// 2. Creates a pipeline layout with the uniform bind group layout defined in
    ///    the `UniformBinding` object.
    /// 3. Configures the vertex state, including the vertex attributes and the
    ///    buffer layout.
    /// 4. Defines the primitive state, including the topology (triangle strip), cull mode, and front face.
    /// 5. Optionally configures a depth-stencil state for depth testing and writing.
    /// 6. Specifies the fragment state, including the blending and render target format.
    /// 7. Assembles the render pipeline with all these configurations.
    ///
    /// # Returns
    ///
    /// A `wgpu::RenderPipeline` instance that can be used for subsequent rendering.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Assuming `device` is an instance of `wgpu::Device`,
    /// // `surface_format` is a valid wgpu::TextureFormat,
    /// // and `uniform` is an instance of `UniformBinding`.
    /// let pipeline = Scene::create_pipeline(&device, surface_format, &uniform);
    /// ```
    fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        uniform: &UniformBinding,
    ) -> wgpu::RenderPipeline {
        // The shader module contains the compiled SPIR-V or WGSL shader code that runs on the GPU.
        //
        // It serves as the container for the vertex and fragment shaders used in the rendering pipeline.
        // The module is created with a provided WGSL shader source and is later referenced in the pipeline
        // configuration to define how the vertex and fragment stages process data.
        //
        // # Purpose
        // - Provides precompiled or compiled-at-runtime shader programs to the GPU.
        // - Acts as an entry point for vertex and fragment shader functions specified in the pipeline.
        //
        // The shaders define how vertex data is transformed and rasterized into fragment data, as well
        // as how fragments are finally processed into pixels on the render target.
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SOURCE)),
        });

        // The pipeline layout defines the structure of resources (such as uniform buffers and
        // textures) available to the shader programs in the rendering pipeline.
        //
        // It organizes how these resources are grouped in bind groups and how they are connected
        // to the GPU pipeline stages, such as vertex and fragment shaders. This layout specifies
        // the binding locations and ensures proper mapping between the shader code and the
        // resources used.
        //
        // # Purpose
        // - Associates bind groups with their layouts for resource availability in shaders.
        // - Establishes a consistent mapping of GPU resources for rendering operations.
        //
        // # Details
        // - The `bind_group_layouts` define the layouts for all bind groups used in the pipeline.
        // - `push_constant_ranges` allows for defining push constants, though it's empty in this case.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&uniform.bind_group_layout],
            push_constant_ranges: &[],
        });

        // Creates and configures a render pipeline, which defines the sequence of operations
        // for rendering, including how vertex and fragment shaders process data, how primitives
        // are assembled, and how the final output is rendered to the screen or a render target.
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            // Configures and creates a render pipeline, which defines how vertex and fragment shaders process
            // the geometry and how the final output is rendered onto the screen or target surface.
            label: None,
            layout: Some(&pipeline_layout), // Specifies the pipeline layout, including bind group layouts.
            vertex: wgpu::VertexState {
                module: &shader_module, // References the compiled vertex shader.
                entry_point: Some("vertex_main"), // Specifies the entry point for the vertex shader.
                buffers: &[Vertex::description(&Vertex::vertex_attributes())], // Defines vertex buffer layout and attributes.
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip, // Specifies how vertices are assembled (triangle strip in this case).
                strip_index_format: Some(wgpu::IndexFormat::Uint32), // Index format used with triangle strips.
                front_face: wgpu::FrontFace::Cw, // Specifies the front-facing direction for culling (clockwise).
                cull_mode: None, // Disables face culling.
                polygon_mode: wgpu::PolygonMode::Fill, // Draws filled polygons.
                conservative: false,
                unclipped_depth: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Renderer::DEPTH_FORMAT, // Specifies the format of the depth buffer for depth testing.
                depth_write_enabled: true, // Enables depth writes to the depth buffer.
                depth_compare: wgpu::CompareFunction::Less, // Configures depth testing to write only closer fragments.
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1, // Anti-aliasing sample count (1 = no anti-aliasing).
                mask: !0, // All samples are active when rasterizing.
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module, // References the compiled fragment shader.
                entry_point: Some("fragment_main"), // Specifies the entry point for the fragment shader.
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format, // Specifies the format of the render target (framebuffer).
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // Enables alpha blending for transparency effects.
                    write_mask: wgpu::ColorWrites::ALL, // Allows writing to all color channels (RGBA).
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        })
    }
}
