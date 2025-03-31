//! # Triangle Renderer Library
//!
//! This library provides the core components for rendering a simple colored triangle using the WebGPU
//! API. It is organized into multiple modules that encapsulate functionality such as application logic,
//! GPU resource management, scene management, and rendering pipeline setup. The library serves as a
//! foundation for learning and experimenting with modern GPU programming techniques.
//!
//! ## Modules
//!
//! - [`app`]: Handles application setup, event loop, and user input integration.
//! - [`renderer`]: Manages the rendering pipeline including shaders, vertex buffers, and GPU commands.
//! - [`gpu`]: Initializes and manages GPU resources such as devices, queues, and surface configuration.
//! - [`scene`]: Encapsulates the scene data including objects, transformations, and lighting.
//! - [`vertex`]: Defines the vertex structure and data used for rendering.
//! - [`uniform_buffer`]: Manages uniform buffer resources, such as transformation matrices.
//! - [`uniform_binding`]: Manages bindings for shaders to access uniform buffer data.
//!
//! ## Constants
//!
//! ### [`INDICES`]
//!
//! This constant defines the order of vertices in the triangle, which is used by the graphics pipeline
//! to assemble the final shape. The order specifies a clockwise winding, a convention often used to
//! specify front-facing triangles in graphics pipelines.
//!
//! ### [`SHADER_SOURCE`]
//!
//! This constant contains the source code for the shader written in WGSL (WebGPU Shading Language).
//! It defines both vertex and fragment shader stages for rendering the triangle, applying transformations
//! using a model-view-projection matrix.
//!
//! ## Features
//!
//! - **WebGPU Integration**: Efficiently leverages GPU resources for high-performance rendering.
//! - **Custom Shaders**: Includes WGSL shaders to demonstrate programmable graphics pipelines.
//! - **Transformations**: Uses uniform buffers for applying model-view-projection transformations.
//! - **Event Handling**: Integrates with input and window handling via `winit`.
//!
//! ## Usage
//!
//! 1. **Initialize the App**: Start by creating an instance of [`App`], which serves as the main entry point.
//! 2. **Setup Render Pipeline**: Use [`Renderer`] to configure shaders, buffers, and other pipeline resources.
//! 3. **Manage GPU Resources**: Use [`Gpu`] to handle device initialization and command execution.
//! 4. **Define the Scene**: Populate a [`Scene`] with vertices, transformations, and other scene data.
//! 5. **Render the Triangle**: Call the render function to draw the triangle using the provided data and pipeline.
//!
//! ## Example
//!
//! ```rust
//! use triangle_renderer::{App, Renderer, Gpu, Scene};
//!
//! // Initialize the application
//! let mut app = App::new();
//!
//! // Set up GPU resources and renderer
//! let gpu = Gpu::new(&app);
//! let mut renderer = Renderer::new(&gpu);
//!
//! // Define the scene
//! let scene = Scene::new();
//!
//! // Main rendering loop
//! app.run(move |event| {
//!     renderer.render(&gpu, &scene);
//! });
//! ```
//!
//! ## Dependencies
//!
//! This library relies on the following external crates:
//!
//! - `wgpu`: For WebGPU support, rendering pipelines, and GPU resource management.
//! - `winit`: For window and event management.
//! - `log` and `env_logger`: For logging.
//! - `nalgebra-glm`: For matrix and vector math operations.
//!
//! ## Notes
//!
//! - Make sure to have the necessary system support for WebGPU before using this library.
//! - Update the uniform buffer appropriately for transformations to ensure proper rendering.
//! - Ensure input vertex data matches the shader's expected layout to avoid runtime errors.

mod app;
mod renderer;
mod gpu;
mod scene;
mod vertex;
mod uniform_buffer;
mod uniform_binding;

use web_time::Duration;

pub use crate::app::App;
pub use crate::renderer::Renderer;
pub use crate::gpu::Gpu;
pub use crate::scene::Scene;
pub use crate::vertex::{Vertex, VERTICES};
pub use crate::uniform_buffer::UniformBuffer;
pub use crate::uniform_binding::UniformBinding;

/// An array of indices defining the order of vertices to draw a triangle.
///
/// This array represents the indices of the `VERTICES` array used by the graphics pipeline
/// to assemble the primitive shape. The order determines which vertices are connected
/// and the winding direction of the triangle.
///
/// - `0`: Refers to the first vertex at index 0 in the `VERTICES` array.
/// - `1`: Refers to the second vertex at index 1 in the `VERTICES` array.
/// - `2`: Refers to the third vertex at index 2 in the `VERTICES` array.
///
/// The indices are in clockwise winding order, which is commonly required to define
/// the front-facing side of the triangle in the graphics pipeline.
pub const INDICES: [u32; 3] = [0, 1, 2]; // Clockwise winding order

/// The source code for the shader written in WGSL (WebGPU Shading Language).
///
/// This shader defines both vertex and fragment stages for rendering a triangle.
/// It uses a uniform buffer object (UBO) for applying transformations to the
/// vertices and outputs color data for the rendered triangle.
///
/// ### Uniform
///
/// The shader declares a uniform buffer:
/// - `ubo`: Contains a single 4x4 matrix named `mvp` (model-view-projection matrix) used to
///   transform vertex positions in the vertex stage.
///
/// ### Vertex Stage
///
/// The vertex shader (`vertex_main`) receives input from the graphics pipeline, defined
/// via `VertexInput`:
/// - `@location(0) position`: The position of the vertex as a 4D vector `[x, y, z, w]`.
/// - `@location(1) color`: The color of the vertex as a 4D vector `[r, g, b, a]`.
///
/// The output of the vertex stage, `VertexOutput`, includes:
/// - `@builtin(position) position`: The transformed position of the vertex.
/// - `@location(0) color`: The color passed through to the fragment shader.
///
/// The vertex shader applies the `mvp` matrix to the vertex position to calculate
/// the transformed position of the vertex.
///
/// ### Fragment Stage
///
/// The fragment shader (`fragment_main`) receives as input the interpolated outputs
/// from the vertex stage:
/// - `@location(0) color`: The interpolated color of the triangle.
///
/// The fragment shader outputs:
/// - `@location(0) vec4<f32>`: The final color of the rendered fragment.
///
/// ### Usage
///
/// This shader can be bound to a graphics pipeline to render a transformed and
/// colored triangle using the vertex data, an `mvp` matrix, and custom color information.
///
/// ### Notes
///
/// - Ensure the uniform buffer is updated to provide the correct `mvp` matrix during rendering.
/// - The inputs to the vertex shader must match the layout of vertices in your vertex buffer.
/// - The outputs of the vertex stage must match the inputs of the fragment stage.
pub const SHADER_SOURCE: &str = include_str!("shader_source.wgsl");
