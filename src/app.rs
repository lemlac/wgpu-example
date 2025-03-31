//! # Application Core (`app.rs`)
//!
//! Main structure and associated implementation for managing the application's state and behavior.
//! The `App` struct serves as the primary entry point for handling platform-specific behavior,
//! GUI state, rendering, and event processing in a WGPU-based application.
//!
//! This file defines the `App` structure along with its fields and methods, enabling seamless
//! integration with the `winit` event loop and cross-platform compatibility for desktop and
//! WebAssembly environments.
//!
//! ## Features and Components
//!
//! The `App` manages several key components integral to the application, including:
//!
//! - **Window Management**: Handles the creation, resizing, and event processing of the application window.
//! - **Renderer**: Manages GPU-based rendering through WGPU, including buffer creation, shaders, and frame updates.
//! - **GUI State**: Integrates the `egui` GUI framework for creating user interfaces and handling input events.
//! - **Platform Compatibility**: Implements synchronous or asynchronous behavior for renderer initialization based on the target platform.
//! - **Frame Timing**: Tracks frame times for animations, rendering optimizations, and consistent performance across platforms.
//!
//! ## Platform-Specific Notes
//!
//! - **Desktop**:
//!   - Uses synchronous initialization of the `Renderer` for fast GPU access during application startup.
//!   - Relies on `pollster` for blocking operations that set up the rendering backend.
//!
//! - **WebAssembly**:
//!   - Uses asynchronous initialization to integrate with browser APIs and support the HTML5 canvas.
//!   - Requires a `oneshot::Receiver` for asynchronously retrieving the renderer instance.
//!   - Logs errors and warnings directly to the browser console for easier debugging.
//!
//! ## Example Usage
//!
//! ```ignore
//! use winit::{event_loop::EventLoop, platform::web::WindowBuilder};
//!
//! let event_loop = EventLoop::new();
//! let mut app = App::default();
//! event_loop.run(move |event, _, control_flow| {
//!     app.handle_event(event, control_flow);
//! });
//! ```
//!
//! ## Struct Overview
//!
//! The `App` struct consists of the following fields:
//!
//! - `window`: Reference-counted pointer to the main application window.
//! - `renderer`: Optional WGPU-based renderer backend for rendering frames.
//! - `gui_state`: Manages integration of the `egui` library for GUI components.
//! - `last_render_time`: Tracks the time of the last frame render for performance monitoring.
//! - `renderer_receiver`: (WebAssembly only) Asynchronous receiver for handling renderer initialization.
//! - `last_size`: Stores the dimensions of the window to handle dynamic resizing.
//!
//! For each field, platform-specific initialization and usage details are included in the struct-level documentation.

// The `wasm_bindgen::prelude::*` is needed to enable seamless interaction between the Rust application 
// and JavaScript for WebAssembly platforms. It provides attributes and macros like `#[wasm_bindgen]`, 
// which make it possible to call Rust functions from JavaScript or vice versa, manage DOM elements, 
// and access browser-specific APIs effectively.
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Importing `Arc` (Atomic Reference Counted) from the standard library, 
// which is used to create thread-safe, reference-counted pointers for shared ownership of data.
// In this context, it ensures safe and efficient sharing of the `Window` instance across threads.
use std::sync::Arc;

// Importing `Instant` from the `web_time` crate, which provides a cross-platform abstraction
// for measuring time. On WebAssembly, it uses high-resolution time from the browser,
// while on other platforms, it falls back to a standard time measurement. This is 
// particularly useful for tracking frame render timings and performance metrics
// in the application.
use web_time::Instant;

// Importing necessary types and traits from the `winit` crate, which is used for window 
// creation and event handling. These include:
// - `ApplicationHandler`: Provides the trait for implementing application-specific event handling logic.
// - `PhysicalSize`: Represents physical dimensions of a window or surface in pixels, used for resizing.
// - `WindowEvent`: Enumerates various events related to the window, such as resizing, focus changes, etc.
// - `Theme`: Allows querying or setting the theme of the application (e.g., Light or Dark mode).
// - `Window`: Represents the main application window used for rendering, GUI, and handling user interactions.
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    window::{Theme, Window},
};

// Importing the `Renderer` type from the local `renderer` module.
// The `Renderer` is the core of the rendering system, responsible for interacting 
// with the WGPU backend to render graphical content. It handles tasks such as
// creating GPU buffers, compiling shaders, and managing frame updates. 
// This is a crucial component for rendering the application's visuals.
use crate::renderer::Renderer;

/// Main application structure for managing the GUI application state.
///
/// The `App` struct implements the `ApplicationHandler` trait to manage
/// the initialization, event handling, and rendering of a WGPU-based GUI
/// application.
///
/// # Fields
///
/// - `window`:
///   An optional reference-counted pointer to the application window.
///   This represents the main window where rendering and GUI interactions occur.
///
/// - `renderer`:
///   An optional WGPU renderer responsible for handling graphical
///   rendering of the application. The renderer is created synchronously
///   for desktop platforms and asynchronously on WebAssembly.
///
/// - `gui_state`:
///   Manages the integration of the `egui` GUI framework with the winit window.
///   It tracks window-scale factors, GUI events, and rendering contexts.
///
/// - `last_render_time`:
///   Tracks the time of the last frame render, used for computing frame timings.
///
/// - `renderer_receiver`: _(WebAssembly only)_
///   A `oneshot::Receiver` that asynchronously receives the WGPU renderer instance
///   after its initialization.
///
/// - `last_size`:
///   Stores the dimensions of the window (width and height) in physical pixels,
///   which are used to resize the rendering surface when necessary.
///
/// - `panels_visible`:
///   A flag to track whether GUI panels are currently visible in the application.
///
/// # Platform-Specific Implementation
///
/// - **Desktop:**
///   - Uses synchronous renderer initialization with the `pollster` crate.
///   - Rendering and GUI updates take place on the main event loop thread.
///
/// - **WebAssembly:**
///   - Performs asynchronous initialization for the renderer and uses browser APIs,
///     such as accessing the canvas element via `wasm_bindgen`.
///   - Sets up a panic hook and initializes logging to the browser console.
///
/// # Usage
///
/// This struct is intended to work as the main application handler, passed to the
/// `winit` event loop. It processes events, manages GUI rendering, and handles
/// platform-specific behavior seamlessly.
///
/// ## Example
///
/// Used as follows:
///
/// ```ignore
/// use winit::{event_loop::EventLoop, platform::web::WindowBuilder};
/// let event_loop = EventLoop::new();
/// let mut app = App::default();
/// event_loop.run(move |event, _, control_flow| {
///     app.handle_event(event, control_flow);
/// });
/// ```
#[derive(Default)]
pub struct App {
    /// The main application window where rendering and GUI interactions occur.
    ///
    /// This is an optional reference-counted pointer (`Arc`) to the `Window` instance created
    /// as part of the application. It is initialized when the application starts or resumes
    /// and is used for handling window events, rendering, and GUI updates.
    ///
    /// On desktop platforms, the window is created synchronously when the application starts.
    /// On WebAssembly, the window references an existing HTML canvas element.
    window: Option<Arc<Window>>,

    /// The WGPU renderer responsible for handling graphical rendering in the application.
    ///
    /// This optional field is used to represent the rendering backend that communicates with the GPU.
    /// The `Renderer` is initialized based on the platform:
    ///
    /// - **Desktop:**
    ///   Rendering is initialized synchronously using the `pollster` crate to wait for the
    ///   asynchronous renderer setup.
    /// - **WebAssembly:**
    ///   Rendering is initialized asynchronously with the use of a `oneshot::Receiver`.
    ///
    /// The renderer handles tasks such as:
    /// - Creating and managing GPU resources (e.g., textures, buffers).
    /// - Handling rendering pipelines, shaders, and drawing calls.
    /// - Managing frame updates and presenting the final frame.
    ///
    /// When this field is `None`, the application is not yet ready for rendering.
    renderer: Option<Renderer>,

    /// Manages the state and integration of the `egui` GUI framework with the `winit` window.
    ///
    /// This optional field stores an instance of `egui_winit::State`, which is responsible
    /// for tracking events, maintaining GUI settings, and providing the necessary context
    /// for rendering egui components.
    ///
    /// - It adapts the egui library to work with platform-specific windowing systems, ensuring
    ///   compatibility with the `winit` event loop and input handling.
    /// - Handles tasks such as:
    ///   - Maintaining window scale factor (pixels per point) and viewport information.
    ///   - Passing platform input events (mouse, keyboard, etc.) to the egui context.
    ///   - Rendering the egui interface onto the screen through the WGPU renderer.
    ///
    /// This field is initialized when the application starts and remains `None` if the GUI
    /// state has not yet been set up.
    gui_state: Option<egui_winit::State>,

    /// Tracks the timestamp of the last render frame.
    ///
    /// This field is an optional `Instant` value that records the time when the
    /// previous frame was rendered. It is primarily used to calculate the time
    /// delta between frames, which can be useful for tasks such as updating
    /// animations or ensuring consistent frame pacing.
    ///
    /// If this value is `None`, it indicates that no frames have been rendered yet
    /// since the application started or resumed.
    last_render_time: Option<Instant>,

    /// A receiver for asynchronously initializing the WGPU renderer on WebAssembly platforms.
    ///
    /// This field is only available when targeting the `wasm32` architecture. It holds a
    /// `oneshot::Receiver` that is used to receive the asynchronously created `Renderer` instance.
    ///
    /// ### Usage
    /// - When the application starts or resumes on WebAssembly, a new channel is created, and the
    ///   `receiver` is stored in this field while the rendering backend is initialized asynchronously.
    /// - Once the initialization completes, the `receiver` yields the `Renderer`, which is then set
    ///   to the `renderer` field.
    ///
    /// ### Platform Integration
    /// - **Desktop:** This field is unused and does not appear since synchronous initialization
    ///   with `pollster` is used instead.
    /// - **WebAssembly:** This allows rendering initialization to integrate seamlessly with
    ///   asynchronous browser-based APIs.
    ///
    /// When this field is `None`, either the platform is not using asynchronous initialization, or
    /// the renderer has already been fully initialized.
    #[cfg(target_arch = "wasm32")]
    renderer_receiver: Option<futures::channel::oneshot::Receiver<Renderer>>,

    /// Tracks the size of the application window during its last update.
    ///
    /// This field stores the dimensions of the window as a tuple of width and height in pixels.
    /// It is updated whenever the window is resized or when the application starts/resumes.
    ///
    /// - **Desktop:** The size is retrieved using the `inner_size` of the `winit` window.
    /// - **WebAssembly:** The size is based on the dimensions of the associated HTML canvas element.
    ///
    /// This value is primarily used for determining the current rendering viewport
    /// and ensuring that the application properly handles window resizing events.
    last_size: (u32, u32),

    /// Indicates whether application panels are visible.
    ///
    /// This boolean field tracks the visibility state of various panels in the application.
    /// Panels may include UI components like menus, sidebars, or overlays that are toggled
    /// on or off during runtime.
    ///
    /// - A value of `true` means that the panels are currently displayed to the user.
    /// - A value of `false` means that the panels are hidden.
    ///
    /// This field is typically used to manage and render UI elements conditionally.
    panels_visible: bool,
}

/// Implements the `ApplicationHandler` trait for `App`, defining how the application
/// responds to specific lifecycle events and manages critical application behaviors.
///
/// The implementation handles the following responsibilities:
/// - Proper initialization and setup of the application when it resumes (e.g., after being started/restarted).
/// - Setting up the application window, including its attributes (such as title or canvas for WebAssembly).
/// - Integration with the `egui` framework for GUI rendering, ensuring the necessary context, state,
///   and window-specific settings are established.
/// - Initializing the WGPU renderer for graphics rendering, working differently depending on the
///   target platform:
///     - For **Desktop platforms:** Synchronous initialization using `pollster`.
///     - For **WebAssembly:** Asynchronous initialization with `oneshot` channels.
///
/// This is a key part of the `App` structure, as it enables:
/// - Creation and maintenance of user-facing application windows.
/// - GUI rendering and event handling using `egui` and `winit`.
/// - Seamless cross-platform support for desktop and WebAssembly platforms.
///
/// By implementing `ApplicationHandler`, this struct becomes capable of responding to application
/// lifecycle events such as starting, resuming, or resizing, enabling the developer to create and manage
/// platform-agnostic applications.
impl ApplicationHandler for App {
    /// Handles the application's resume event, performing setup and initialization tasks.
    ///
    /// This method is called when the application is resumed or started. It sets up the application
    /// window, handles platform-specific configurations (e.g., WebAssembly canvas setup), and ensures
    /// that all required resources for rendering and GUI integration are properly initialized.
    ///
    /// ### Desktop:
    /// - Creates a window with default attributes and sets a default title.
    /// - Initializes the rendering backend synchronously with `pollster`.
    /// - Sets up `egui` state for GUI rendering.
    ///
    /// ### WebAssembly:
    /// - Retrieves the HTML canvas element by its ID and adjusts the application for the browser environment.
    /// - Asynchronously initializes the rendering backend using oneshot channels.
    ///
    /// ### Behavior:
    /// - If this is the first window being created for the application, additional tasks, such as
    ///   setting up the `egui` context and GUI state, are performed.
    /// - On WebAssembly platforms, platform-specific hooks and loggers are configured to integrate seamlessly
    ///   with the browser environment.
    ///
    /// ### Parameters:
    /// - `event_loop`: A reference to the `ActiveEventLoop` that is used to create and manage application windows.
    ///
    /// This method ensures that the application is ready to render frames and interact with the user
    /// across different platforms (desktop and WebAssembly).
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Represents the attributes used to create or configure an application window.
        //
        // This field is initialized with default attributes and can be further customized
        // to define platform-specific window properties. The attributes dictate the behavior
        // and appearance of the application window during its creation.
        //
        // ### Examples:
        // - On **Desktop platforms**, the `title` can be set to describe the application window.
        // - On **WebAssembly**, the canvas element can be defined for the application, ensuring the
        //   rendering surface is correctly tied to the HTML document.
        //
        // This variable plays a key role in the cross-platform support capabilities of this application,
        // allowing platform-specific customization via feature flags or attributes extensions.
        let mut attributes = Window::default_attributes();

        #[cfg(not(target_arch = "wasm32"))]
        {
            attributes = attributes.with_title("Standalone Winit/Wgpu Example");
        }

        // Represents the width of the HTML canvas element when running on WebAssembly.
        //
        // This variable is used to determine the initial width of the application rendering surface
        // in the browser environment. It is retrieved dynamically from the canvas element's properties
        // and ensures that the canvas dimensions are accurately reflected in the application state.
        //
        // ### Usage:
        // - The value is initialized when the application retrieves the canvas element by its ID.
        // - It is used to set up the `last_size` field, helping the application adapt rendering logic
        //   based on the actual canvas dimensions.
        //
        // ### Platform-specific:
        // - On **WebAssembly**, this value is critical for ensuring that the rendering surface matches
        //   the size of the HTML canvas.
        // - On **Desktop platforms**, this variable is not used since the window size is managed by `winit`.
        #[allow(unused_assignments)]
        #[cfg(target_arch = "wasm32")]
        let mut canvas_width = 0;

        // Represents the height of the HTML canvas element when running on WebAssembly.
        //
        // This variable is used to determine the initial height of the application rendering surface
        // in the browser environment. Similar to `canvas_width`, it is dynamically retrieved from the
        // canvas element's properties, ensuring the application state accurately reflects the canvas dimensions.
        //
        // ### Usage:
        // - The value is initialized when the application retrieves the canvas element by its ID.
        // - It is used to set up the `last_size` field, allowing the application to correctly scale
        //   its rendering logic based on the actual canvas dimensions.
        //
        // ### Platform-specific:
        // - On **WebAssembly**, this value is critical for integrating with the browser environment
        //   and ensuring that the rendering surface scales correctly.
        // - On **Desktop platforms**, this variable is not relevant because the window size is managed
        //   by the desktop-specific APIs.
        #[allow(unused_assignments)]
        #[cfg(target_arch = "wasm32")]
        let mut canvas_height = 0;

        // Platform-specific configuration for WebAssembly:
        // This block retrieves the HTML canvas element using its ID "canvas" and sets the application's
        // window attributes with this canvas. It also initializes the `last_size` field to match the
        // canvas dimensions (width and height). This setup ensures that the application correctly adapts
        // to the browser environment when running on WebAssembly. The use of `WindowAttributesExtWebSys`
        // allows the configuration of the canvas element to be integrated into the `winit` window system.
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;

            // Represents the HTML canvas element used as the rendering surface when running on WebAssembly.
            //
            // This variable references the HTML canvas element retrieved from the DOM via its unique ID ("canvas").
            // The canvas serves as the main rendering surface for the application and is directly tied
            // to the browser environment. This connection allows the application to seamlessly integrate
            // its rendering logic with the web platform.
            //
            // ### Usage:
            // - The canvas is retrieved dynamically using the `web_sys` API and configured for rendering.
            // - It is essential for initializing the application window and graphics context, especially
            //   on WebAssembly targets.
            //
            // ### Platform-specific:
            // - On **WebAssembly**, this variable is critical for associating the application rendering
            //   backend with the browser's canvas element.
            // - On **Desktop platforms**, the canvas is not applicable as the rendering is handled by
            //   the `winit` window system.
            //
            // ### Notes:
            // - The `canvas` element must exist in the HTML document with the ID "canvas" for the application
            //   to initialize correctly.
            // - The dimensions of the canvas are dynamically retrieved to synchronize it with the application's
            //   rendering state.
            //
            // ### Examples:
            // - The `canvas.width()` and `canvas.height()` are used to initialize the size of the rendering
            //   surface in the browser.
            // - The retrieved canvas is passed to the `WindowAttributes` configuration using the
            //   `attributes.with_canvas(Some(canvas))` method.
            let canvas = wgpu::web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .unwrap()
                .dyn_into::<wgpu::web_sys::HtmlCanvasElement>()
                .unwrap();
            canvas_width = canvas.width();
            canvas_height = canvas.height();
            self.last_size = (canvas_width, canvas_height);
            attributes = attributes.with_canvas(Some(canvas));
        }

        // Checks if creating a window using the given attributes was successful.
        // If successful, it proceeds with window setup for GUI rendering and renderer initialization.
        if let Ok(window) = event_loop.create_window(attributes) {
            // Attempts to create a new application window using the specified attributes.
            // If the window creation is successful, the block initializes the necessary
            // application state for GUI rendering and graphics processing. This includes:
            // - Assigning the created window handle to the `self.window` field.
            // - Performing setup tasks if this is the first window being created, such as:
            //     - Initializing `egui` context and state for GUI integration.
            //     - Setting the application's initial size and scaling behaviors based on the platform.
            //     - Configuring the rendering backend:
            //       - On desktop platforms, synchronously using `pollster` for WGPU initialization.
            //       - On WebAssembly, asynchronously initializing WGPU with an `oneshot` channel.
            // - Logging and panic hook setup (specific to WebAssembly).
            // The `self.window` field is populated with the created window handle, and other related
            // state variables like `self.renderer` and `self.gui_state` are initialized to ensure
            // the application is ready for rendering and GUI interactions.

            // A boolean flag indicating whether this is the first window being created for the application.
            //
            // This variable is used to determine if additional tasks, such as initializing the GUI state,
            // rendering context, and other platform-specific configurations, need to be performed. These
            // tasks are only executed when the first window is being created, ensuring proper application setup.
            //
            // ### Behavior:
            // - `true`: This is the first window being created. Initialization tasks will be executed.
            // - `false`: This is not the first window. Skips the initialization tasks.
            //
            // The value is derived by checking if the `self.window` field is `None` at the time
            // of window creation.
            let first_window_handle = self.window.is_none();

            // Represents the handle to the created application window.
            //
            // This variable holds a reference-counted `Arc` to the window that has been successfully created
            // using the `winit` event loop. The stored handle allows the application to perform operations
            // on the window, such as resizing, handling inputs, or adapting rendering configurations.
            //
            // ### Key Roles:
            // - Ensures a strong reference to the created window is maintained in the application state.
            // - Facilitates interaction with the underlying platform's window management functions via `winit`.
            //
            // ### Notes:
            // - The `Arc` wrapper allows the handle to be shared between multiple components or threads
            //   if necessary.
            // - This field is updated only if window creation is successful, ensuring the application
            //   state remains consistent.
            //
            // ### Examples:
            // - On desktop platforms, the `window_handle` is used to query dimensions or manage window properties.
            // - For WebAssembly, the handle integrates with the HTML canvas for browser-based rendering.
            //
            // Initialized with:
            // ```rust
            // let window_handle = Arc::new(window);
            // ```
            let window_handle = Arc::new(window);

            self.window = Some(window_handle.clone());
            if first_window_handle {
                // Checks if this is the first time a window is being created for the application.
                // If it is, performs several initialization steps for the application's state:
                // - Creates and sets up an `egui` context and GUI state for rendering.
                // - Configures platform-specific settings, such as updating the size and scaling for GUI.
                // - Initializes the rendering backend:
                //     - On desktop platforms, it uses a synchronous approach with `pollster`.
                //     - On WebAssembly, it sets up an asynchronous initialization and logging.
                // - Stores the `window_handle` in `self.window` and initializes other related state
                //   fields, like `self.renderer` and `self.gui_state`, to prepare for GUI rendering
                //   and frame updates.

                // Represents the `egui::Context` instance for managing GUI rendering and state.
                //
                // This context serves as the central hub for all GUI-related operations and configurations
                // within the application. It encapsulates functionality for:
                //
                // - Rendering GUI components.
                // - Tracking and handling input events (e.g., keyboard, mouse) for GUI interactions.
                // - Managing persistent state across frames (e.g., window positions, user interactions).
                //
                // ### Key Roles:
                // - Allows seamless integration of `egui` with the application's rendering and input pipeline.
                // - Acts as the primary interface for drawing the GUI during frame updates.
                //
                // ### Notes:
                // - On WebAssembly targets, the context is configured to account for the `scale_factor`
                //   from the platform to ensure proper rendering scaling for high-DPI environments.
                // - This instance is initialized with `egui::Context::default()` and can be customized
                //   further as the application evolves.
                //
                // ### Examples:
                // - Setting the application's theme (light or dark mode).
                // - Drawing widgets such as buttons, sliders, or text boxes.
                // - Managing platform-specific behaviors related to screen scaling or input systems.
                //
                // The `gui_context` is a central component for initializing and managing all subsequent
                // GUI-related stages in the application.
                let gui_context = egui::Context::default();

                #[cfg(not(target_arch = "wasm32"))]
                {
                    // Represents the inner dimensions of the created window, in physical pixels.
                    //
                    // This variable is used to determine and store the width and height of the application window
                    // when it is initialized. The dimensions are critical for various aspects of the application, such as:
                    //
                    // - Setting up initial rendering configurations.
                    // - Adjusting GUI scaling and layout based on the window size.
                    // - Ensuring the correct handling of DPI scaling.
                    //
                    // ### Notes:
                    // - On desktop platforms, the dimensions are retrieved using the `inner_size()` method of the `window_handle`.
                    // - On WebAssembly, this step may differ as it requires adapting the rendering pipeline to the current canvas size.
                    //
                    // ### Example Usage:
                    // ```rust
                    // let inner_size = window_handle.inner_size();
                    // ```
                    let inner_size = window_handle.inner_size();
                    self.last_size = (inner_size.width, inner_size.height);
                }

                // This code block handles platform-specific initialization for desktop and WebAssembly targets.
                //
                // - On desktop platforms (`not(target_arch = "wasm32")`):
                //   - Retrieves the window's inner size to store its width and height, which are
                //     essential for initializing state like rendering configurations and GUI scaling.
                //
                // - On WebAssembly platforms (`target_arch = "wasm32"`):
                //   - Sets the pixels per point in the `egui::Context` to correctly account for the
                //     `scale_factor`, ensuring proper scaling of the GUI.
                //
                // These steps ensure the application adapts to the platform's specific requirements
                // for accurate window size, DPI scaling, and rendering behaviors.
                #[cfg(target_arch = "wasm32")]
                {
                    gui_context.set_pixels_per_point(window_handle.scale_factor() as f32);
                }

                // Represents the unique identifier for the viewport associated with the `egui::Context`.
                //
                // This identifier is used to link the GUI context to a specific rendering surface or window.
                // It is essential for ensuring that GUI elements are drawn within the appropriate viewport,
                // particularly when dealing with multiple windows or dynamic scaling configurations.
                //
                // ### Key Characteristics:
                // - Acts as a handle to the rendering target for GUI operations.
                // - Ensures the correct association between the GUI context and the window dimensions or DPI scaling.
                //
                // ### Usage Notes:
                // - In most applications with a single viewport, this ID directly maps to the primary window.
                // - In multi-window configurations, it helps associate GUI contexts with their respective render targets.
                //
                // ### Platform-Specific Behavior:
                // - On desktop platforms, this ID is stable and directly corresponds to the `winit` or windowing system's identifiers.
                // - On WebAssembly, it adapts to the platform specifics, ensuring proper scaling and viewport positioning.
                let viewport_id = gui_context.viewport_id();

                // Represents the state of the `egui` integration with the `winit` windowing system.
                //
                // This variable encapsulates the configuration and behavior for bridging `egui` with `winit`,
                // ensuring proper handling of GUI input events, rendering, and DPI scaling. The `State` struct
                // manages the interaction between the platform's event loop and `egui`'s GUI rendering pipeline.
                //
                // ### Key Responsibilities:
                // - Forwarding user input events (e.g., mouse, keyboard) from the platform to `egui`.
                // - Managing DPI scaling (pixels per point) to ensure the GUI is correctly scaled on different devices.
                // - Bridging the gap between the window's lifecycle (e.g., resize, redraw) and `egui`'s state updates.
                //
                // ### Notes:
                // - The state is platform-sensitive, adapting its behavior based on whether the application is running
                //   on desktop or WebAssembly.
                // - DPI scaling factors (`pixels_per_point`) are explicitly set for accurate rendering on high-DPI displays.
                //
                // ### Customization Options:
                // - `theme`: Allows setting visual styles like dark mode or light mode.
                // - `input`: Configures how input events are handled or transformed for `egui`.
                //
                // ### Example Use Case:
                // This instance is commonly used to supply the GUI configuration and manage inputs throughout
                // the lifecycle of a frame:
                // ```rust
                // let gui_state = egui_winit::State::new(
                //     gui_context,
                //     viewport_id,
                //     &window_handle,
                //     Some(window_handle.scale_factor() as _),
                //     Some(Theme::Dark),
                //     None,
                // );
                // ```
                let gui_state = egui_winit::State::new(
                    gui_context,
                    viewport_id,
                    &window_handle,
                    Some(window_handle.scale_factor() as _),
                    Some(Theme::Dark),
                    None,
                );

                // Represents the width of the window's inner size in physical pixels.
                //
                // This value is used to configure rendering surface dimensions and GUI layouts.
                // Adjustments to this value are necessary when the window is resized, ensuring
                // that the graphics pipeline properly updates the rendering target.
                //
                // ### Platform-Specific Notes:
                // - On non-WebAssembly platforms, this value is retrieved directly from the
                //   `winit` window's `inner_size()`.
                // - On WebAssembly platforms, specific handling via canvas dimensions is typically required.
                #[cfg(not(target_arch = "wasm32"))]
                let width: u32 = window_handle.inner_size().width;

                // Represents the height of the window's inner size in physical pixels.
                //
                // Similar to `width`, this value ensures proper scaling and accurate representation
                // of the GUI and rendering surface based on the window's current dimensions.
                //
                // ### Platform-Specific Notes:
                // - Retrieved from the `inner_size()` method on non-WebAssembly platforms.
                // - Managed explicitly for WebAssembly platforms where the canvas size is used.
                #[cfg(not(target_arch = "wasm32"))]
                let height: u32 = window_handle.inner_size().height;

                #[cfg(not(target_arch = "wasm32"))]
                {
                    env_logger::init();

                    // Represents the primary renderer responsible for handling graphics rendering.
                    //
                    // This variable is initialized during application startup and performs key tasks
                    // such as managing the swap chain and rendering pipeline, ensuring the graphical
                    // output matches the application's current state and window configuration.
                    //
                    // ### Key Responsibilities:
                    // - Creates and manages the rendering pipeline.
                    // - Handles resizing of the rendering surface when the window dimensions change.
                    // - Ensures smooth rendering of frames in synchronization with the event loop.
                    //
                    // ### Platform-Specific Notes:
                    // - On desktop platforms (non-WebAssembly), the renderer is initialized immediately
                    //   using a blocking async call to ensure it's ready before entering the main loop.
                    // - On WebAssembly, the renderer setup is asynchronous, leveraging promises
                    //   (`futures`), given the platform's constraints regarding graphics initialization.
                    //
                    // ### Example Use Case:
                    // The renderer is directly utilized for drawing frames and updating the graphics
                    // pipeline as required:
                    // ```rust
                    // if let Some(ref renderer) = self.renderer {
                    //     renderer.render_frame(&gui_data);
                    // }
                    // ```
                    //
                    // The renderer ensures proper graphics output and integrates seamlessly with the
                    // GUI state and the platform's rendering pipeline.
                    let renderer = pollster::block_on(async move {
                        Renderer::new(window_handle.clone(), width, height).await
                    });
                    self.renderer = Some(renderer);
                }

                #[cfg(target_arch = "wasm32")]
                {
                    // The `sender` is part of a one-shot channel used to communicate the renderer instance
                    // from the asynchronous setup task to the main application state. Once the renderer is
                    // successfully created, the `sender` sends the instance, and the corresponding `receiver`
                    // retrieves it to complete the initialization process.
                    //
                    // ### Key Responsibilities:
                    // - Acts as the sending end of the channel, ensuring the renderer instance is passed
                    //   to the main application logic.
                    // - This ensures proper asynchronous handling for WebAssembly platforms where the renderer
                    //   setup is non-blocking and occurs in a background task.
                    //
                    // ### Notes:
                    // - On successful renderer creation, `sender.send()` is used to transmit the instance.
                    // - If the `sender` is dropped before sending, it signifies an error in renderer creation.
                    //
                    // ---
                    //
                    // The `receiver` is part of the one-shot channel used to retrieve the renderer instance
                    // created in an asynchronous task. It waits for the `sender` to transmit the renderer
                    // and facilitates its assignment to the appropriate field in the application state.
                    //
                    // ### Key Responsibilities:
                    // - Waits for the renderer instance sent by the `sender`.
                    // - Ensures the main application logic remains synchronized with the completion of
                    //   the asynchronous renderer initialization process.
                    //
                    // ### Notes:
                    // - On successful reception, the renderer instance is set up for the application.
                    // - If the `sender` fails or drops before sending the renderer, the `receiver` will
                    //   return an error, which is handled to log the failure gracefully.
                    let (sender, receiver) = futures::channel::oneshot::channel();
                    self.renderer_receiver = Some(receiver);
                    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
                    console_log::init().expect("Failed to initialize logger!");
                    log::info!("Canvas dimensions: ({canvas_width} x {canvas_height})");
                    wasm_bindgen_futures::spawn_local(async move {
                        // Represents the primary renderer responsible for managing graphical output and ensuring the rendering pipeline aligns with the application's state.
                        //
                        // This instance is initialized as part of the application setup. It is responsible for core tasks such as creating shaders, setting up the swap
                        // chain, and synchronizing rendering operations with the windowing system.
                        //
                        // ### Responsibilities:
                        // - Manage the rendering pipeline and swap chains.
                        // - Adapt to window size changes by reinitializing rendering resources.
                        // - Render frames in coordination with GUI interactions and the application's main event loop.
                        //
                        // ### Notes on Asynchronous Initialization:
                        // - **Desktop Platforms:** The renderer is initialized using a blocking async call (`pollster::block_on`) to ensure readiness before the main event loop starts.
                        // - **WebAssembly:** Due to asynchronous restrictions on WebAssembly, the renderer is set up non-blockingly through an async task, leveraging `wasm_bindgen_futures::spawn_local`.
                        //
                        // ### Error Handling:
                        // - On WebAssembly, failure to create or send the renderer instance is logged as an error using the `log` crate.
                        //
                        // ### Example Use:
                        // After initialization, the renderer is used to render frames:
                        // ```rust
                        // if let Some(renderer) = &self.renderer {
                        //     renderer.render_frame(&gui_data);
                        // }
                        // ```
                        let renderer =
                            Renderer::new(window_handle.clone(), canvas_width, canvas_height).await;
                        if sender.send(renderer).is_err() {
                            log::error!("Failed to create and send renderer!");
                        }
                    });
                }

                self.gui_state = Some(gui_state);
                self.last_render_time = Some(Instant::now());
            }
        }
    }

    /// Handles window events coming from the winit event loop.
    ///
    /// This function processes various window-related events, such as resizing, keyboard input,
    /// GUI interactions, close requests, and frame redraw requests. It ensures seamless integration
    /// between the application state and the underlying GUI and rendering systems. Additionally, it
    /// provides platform-specific handling for WebAssembly targets where asynchronous renderer setup
    /// is required.
    ///
    /// # Parameters
    /// - `event_loop`: Reference to the active event loop, used to control application state (e.g., exiting).
    /// - `_window_id`: The ID of the window that triggered the event. Currently unused.
    /// - `event`: The `WindowEvent` instance containing details about the event that occurred.
    ///
    /// # Behavior
    /// - On WebAssembly, checks if the asynchronous renderer initialization has completed
    ///   and assigns the renderer to the appropriate field if ready.
    /// - Makes an early return if any of the required application state components (`gui_state`,
    ///   `renderer`, `window`, `last_render_time`) are missing.
    /// - Routes events to the GUI state. If the event is consumed by the GUI,
    ///   it does not handle it further.
    /// - Intercepts certain events for additional processing:
    ///   - `KeyboardInput`: Closes the application if the Escape key is pressed.
    ///   - `Resized`: Adjusts the renderer's surface size to match the new dimensions.
    ///   - `CloseRequested`: Exits the application when a close request is received.
    ///   - `RedrawRequested`: Triggers GUI rendering and updates the renderer with
    ///     the current frame data.
    ///
    /// # Example
    /// In the case of a window resize event, the function logs the new size and ensures
    /// the renderer adjusts accordingly:
    ///
    /// ```ignore
    /// WindowEvent::Resized(PhysicalSize { width, height }) => {
    ///     log::info!("Resizing renderer surface to: ({width}, {height})");
    ///     renderer.resize(width, height);
    ///     self.last_size = (width, height);
    /// }
    /// ```
    ///
    /// The function ensures that the application responds gracefully to user interactions
    /// and system events and highlights the modular handling of events based on type.
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        // Handles platform-specific logic for WebAssembly (`wasm32`) environments.
        // The block checks whether the asynchronous renderer initialization has been completed
        // by attempting to receive the renderer instance from the `renderer_receiver` channel.
        // If the renderer is successfully received, it is assigned to the `self.renderer` field,
        // and the receiver is set to `None` to indicate that no further waiting is required.
        // This ensures the application correctly initializes the renderer in an asynchronous
        // manner, which is required for WebAssembly environments.
        #[cfg(target_arch = "wasm32")]
        {
            let mut renderer_received = false;
            if let Some(receiver) = self.renderer_receiver.as_mut() {
                if let Ok(Some(renderer)) = receiver.try_recv() {
                    self.renderer = Some(renderer);
                    renderer_received = true;
                }
            }
            if renderer_received {
                self.renderer_receiver = None;
            }
        }

        // Destructures and checks if all necessary components of the application state
        // (`gui_state`, `renderer`, `window`, `last_render_time`) are available.
        // If any of them is missing, the function exits early. This ensures that
        // subsequent operations only proceed when the application is in a valid and
        // fully initialized state.
        let (Some(gui_state), Some(renderer), Some(window), Some(last_render_time)) = (
            self.gui_state.as_mut(),
            self.renderer.as_mut(),
            self.window.as_ref(),
            self.last_render_time.as_mut(),
        ) else {
            return;
        };

        // Receive gui window event
        if gui_state.on_window_event(window, &event).consumed {
            return;
        }

        // If the gui didn't consume the event, handle it
        match event {
            WindowEvent::KeyboardInput {
                event:
                winit::event::KeyEvent {
                    physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                    ..
                },
                ..
            } => {
                // Exit by pressing the escape key

                if matches!(key_code, winit::keyboard::KeyCode::Escape) {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(PhysicalSize { width, height }) => {
                // Handles the `Resized` event, which is triggered when the window size changes.
                // It logs the new width and height dimensions, updates the renderer's surface
                // to match the new size, and stores the new dimensions in `self.last_size`.

                log::info!("Resizing renderer surface to: ({width}, {height})");
                renderer.resize(width, height);
                self.last_size = (width, height);
            }
            WindowEvent::CloseRequested => {
                // Handles the `CloseRequested` event, which is emitted when the user attempts to close the window.
                // This is typically triggered when clicking on the window's close button.
                // Logs a message indicating the close request and calls `event_loop.exit()` to terminate
                // the application cleanly and exit the event loop.

                log::info!("Close requested. Exiting...");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Handles the `RedrawRequested` event, which is triggered by the
                // system or `window.request_redraw()` when a redraw of the window's
                // content is required. This block performs the following tasks:
                // 1. Calculates the time elapsed since the last frame render.
                // 2. Prepares the GUI input and processes it within the Egui context.
                // 3. Conditionally shows various interactive GUI panels (like top, side, and bottom panels)
                //    and a main window based on the `self.panels_visible` state.
                // 4. Finalizes the Egui GUI pass, gets the drawing commands (paint jobs),
                //    and updates any necessary platform GUI behavior.
                // 5. Updates the renderer by passing in frame data, including the screen
                //    size, processed GUI data, and elapsed time since the last render.
                // This ensures smooth rendering of the application interface, including the GUI and graphics.

                // The current instant in time, captured using the `Instant::now()` function.
                // This is used to measure the time elapsed since the last frame render
                // and to calculate the delta time for smooth animations and updates within the application.
                let now = Instant::now();

                // `delta_time` represents the time duration that has elapsed since the last frame was rendered.
                //
                // This value is calculated as the difference between the current time (`now`) and the timestamp
                // stored in `last_render_time`. It is used to ensure smooth and consistent animations,
                // physics calculations, and other time-dependent updates within the application.
                //
                // Measuring frame delta time is essential for ensuring frame-independent behavior, allowing
                // animations and interactions to maintain consistent timing regardless of rendering speed or performance.
                //
                // Units: `delta_time` is a `std::time::Duration`, representing the elapsed time in seconds
                // and nanoseconds.
                let delta_time = now - *last_render_time;
                *last_render_time = now;

                // `gui_input` contains the input data received from the window,
                // such as pointer events, keyboard events, and other UI-related inputs.
                // This data is taken from the window and passed to the Egui context in order to
                // handle and process user interactions for rendering and updating GUI components.
                let gui_input = gui_state.take_egui_input(window);
                gui_state.egui_ctx().begin_pass(gui_input);

                // The `title` variable contains the title of the application window.
                //
                // This title is determined by compile-time configurations, which allow conditional compilation
                // for different platforms and features. For example:
                //
                // - When not targeting `wasm32`, the title defaults to "Rust/Wgpu".
                // - When the `webgpu` feature is enabled, the title is set to "Rust/Wgpu/Webgpu".
                // - When the `webgl` feature is enabled, the title is set to "Rust/Wgpu/Webgl".
                //
                // This ensures the application title accurately reflects the platform or feature in use.
                #[cfg(not(target_arch = "wasm32"))]
                let title = "Rust/Wgpu";

                #[cfg(feature = "webgpu")]
                let title = "Rust/Wgpu/Webgpu";

                #[cfg(feature = "webgl")]
                let title = "Rust/Wgpu/Webgl";

                if self.panels_visible {
                    // Displays the top, left, right, and bottom panels if `self.panels_visible` is true.
                    // Each panel contains interactive GUI elements such as headings and buttons,
                    // which can trigger specific actions when clicked (e.g., logging button clicks).
                    // This block defines the layout and functionality for these GUI panels.

                    // Creates a top panel using `egui::TopBottomPanel` with the identifier "top" and renders its content.
                    // The `show` method is used to build and display the GUI elements defined within the closure (`|ui|`).
                    // Inside the closure, the panel is populated with horizontal navigation options labeled "File" and "Edit".
                    egui::TopBottomPanel::top("top").show(gui_state.egui_ctx(), |ui| {
                        ui.horizontal(|ui| {
                            ui.label("File");
                            ui.label("Edit");
                        });
                    });

                    // Creates a left-side panel using `egui::SidePanel` with the identifier "left" and renders its content.
                    // The `show` method defines the layout and interactive elements inside the panel through a closure (`|ui|`).
                    // Within this closure, a heading labeled "Scene Explorer" is displayed.
                    // Additionally, a button labeled "Click me!" is rendered, and when clicked, it logs a message using the `log` crate.
                    egui::SidePanel::left("left").show(gui_state.egui_ctx(), |ui| {
                        ui.heading("Scene Explorer");
                        if ui.button("Click me!").clicked() {
                            log::info!("Button clicked!");
                        }
                    });

                    // Creates a right-side panel using `egui::SidePanel` with the identifier "right" and renders its content.
                    // The `show` method is used to define the panel's layout and interactive elements within a closure (`|ui|`).
                    // Inside this closure, a heading labeled "Inspector" is displayed.
                    // Additionally, a button labeled "Click me!" is rendered, and when clicked, a message is logged using the `log` crate.
                    egui::SidePanel::right("right").show(gui_state.egui_ctx(), |ui| {
                        ui.heading("Inspector");
                        if ui.button("Click me!").clicked() {
                            log::info!("Button clicked!");
                        }
                    });

                    // Creates a bottom panel using `egui::TopBottomPanel` with the identifier "bottom" and renders its content.
                    // The `show` method is used to define the layout and interactive elements within the panel through a closure (`|ui|`).
                    // Inside this closure, a heading labeled "Assets" is displayed.
                    // Additionally, a button labeled "Click me!" is rendered, and when clicked, a message is logged using the `log` crate.
                    egui::TopBottomPanel::bottom("bottom").show(gui_state.egui_ctx(), |ui| {
                        ui.heading("Assets");
                        if ui.button("Click me!").clicked() {
                            log::info!("Button clicked!");
                        }
                    });
                }

                // Renders a dynamic, interactive window using `egui::Window`.
                // This window displays a checkbox that toggles the visibility of GUI panels
                // (controlled by `self.panels_visible`). The title of the window is determined
                // by compile-time settings, adjusting based on the platform or features.
                egui::Window::new(title).show(gui_state.egui_ctx(), |ui| {
                    ui.checkbox(&mut self.panels_visible, "Show Panels");
                });

                // This let statement creates an interactive GUI window using `egui::Window`.
                //
                // - The `new` method initializes the window with a title, which is derived
                //   from the `title` variable determined earlier based on compile-time configurations.
                // - The `show` method renders the window within the Egui context (`gui_state.egui_ctx()`),
                //   providing a closure (`|ui|`) that specifies the GUI elements and interactions within the window.
                // - In this particular case, a checkbox is added to toggle the visibility of GUI panels
                //   by modifying the `self.panels_visible` property.
                //
                // This mechanism allows for dynamic and interactive user interfaces within the 3D application.
                let egui_winit::egui::FullOutput {
                    textures_delta,
                    shapes,
                    pixels_per_point,
                    platform_output,
                    ..
                } = gui_state.egui_ctx().end_pass();

                gui_state.handle_platform_output(window, platform_output);

                // A collection of painting jobs generated by the Egui framework
                // after tessellating the shapes defined in the GUI context.
                //
                // The `paint_jobs` variable contains instructions for rendering
                // GUI shapes and elements. These jobs are produced by the tessellation
                // process, which converts the GUI's high-level visual elements (e.g.,
                // labels, buttons, panels) into a set of low-level graphical primitives
                // (triangles and vertices) to be rendered by the GPU.
                //
                // The jobs refer to specific textures, vertex coordinates, and
                // other properties required for displaying the GUI accurately
                // on the screen. It also accounts for proper scaling through the
                // `pixels_per_point` parameter.
                //
                // These painting jobs are later passed to the renderer for processing
                // and drawing in the final frame.
                let paint_jobs = gui_state.egui_ctx().tessellate(shapes, pixels_per_point);

                // Represents the display parameters needed for rendering a graphical frame on the screen.
                //
                // The `screen_descriptor` variable contains information about the size of the rendering surface
                // and the scaling factor to account for high-DPI displays. This information is essential for
                // ensuring that graphical elements (both 3D content and GUI) are rendered at the correct size
                // and position on the screen.
                //
                // # Fields
                //
                // - `size_in_pixels`: A 2-element array representing the width and height of the rendering surface
                //   in physical pixels. This is derived from the `self.last_size` property, which holds the latest
                //   dimensions of the window.
                // - `pixels_per_point`: A floating-point value representing the scaling factor to account for
                //   high-DPI displays (e.g., retina displays). This is fetched using the `scale_factor` method
                //   of the `window` object.
                let screen_descriptor = {
                    let (width, height) = self.last_size;
                    egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [width, height],
                        pixels_per_point: window.scale_factor() as f32,
                    }
                };

                renderer.render_frame(screen_descriptor, paint_jobs, textures_delta, delta_time);
            }
            _ => (),
        }

        window.request_redraw();
    }
}
