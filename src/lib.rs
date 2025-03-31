//! # Application Handler for a WGPU-based GUI Application
//!
//! This file implements the main application logic for a GUI program using Winit and WGPU. 
//! The application is designed to support both desktop and WebAssembly (WASM) platforms 
//! with conditional compilation for platform-specific behavior.
//!
//! ## File Purpose
//! - Manages the application lifecycle through the `ApplicationHandler` trait implementation.
//! - Handles platform-specific initialization, including setting up a WGPU renderer and 
//!   integrating it with an EGUI-based GUI.
//! - Processes events such as window resizing, input handling, and rendering updates.
//!
//! ## Key Components
//!
//! ### `App` Structure
//! - Represents the main application state.
//! - Contains fields for managing the window, renderer, GUI state, and rendering timing:
//!   - `window`: A reference to the application window.
//!   - `renderer`: Handles graphical rendering using WGPU.
//!   - `gui_state`: Manages EGUI integration with window events.
//!   - `last_size`: Keeps track of the last known window size.
//!   - `last_render_time`: Tracks the last frame render time.
//!   - `panels_visible`: Tracks whether GUI panels are visible.
//!
//! ### Initialization (`resumed` Method)
//! - Creates and initializes the application window with platform-specific attributes.
//! - On desktop platforms, the WGPU renderer is created synchronously.
//! - On WebAssembly, the initialization is asynchronous, and a panic hook and logging are set up.
//! - Integrates EGUI, configuring the GUI settings and attaching it to the window.
//!
//! ### Event Handling (`window_event` Method)
//! - Manages application events dispatched from the windowing system:
//!   - Keyboard input: Allows closing the application with the `Escape` key.
//!   - Window resizing: Resizes the rendering surface accordingly.
//!   - Close requests: Closes the application cleanly.
//!   - Redraw requests: Invokes rendering and GUI updates for each frame.
//!
//! ### Platform-Specific Logic
//! - **Desktop:**
//!   - Synchronous initialization of the renderer.
//!   - Uses blocking operations (like `pollster::block_on`) for setup.
//! - **WebAssembly:**
//!   - Asynchronous renderer setup using `wasm_bindgen` and `futures`.
//!   - Uses browser-specific APIs, like the DOM for canvas element retrieval.
//!   - Logs errors and warnings to the browser console.
//!
//! ## Example Usage
//! This code is intended to be part of an application entry point, where `App` is used as the 
//! main handler for the `winit` event loop, managing input, rendering, and the GUI system.

// The `wasm_bindgen::prelude::*` is a wildcard import from the `wasm-bindgen` crate,
// which provides the necessary tools to interact between WebAssembly (WASM) and
// JavaScript. This includes macros and traits for exporting Rust functions
// to JavaScript or importing JavaScript functions into Rust.
//
// Key items it brings into scope:
// - `#[wasm_bindgen]` attribute: Used to mark functions, structures, or impls
//   for JavaScript interoperability.
// - Types like `JsValue`: Represent values passed between Rust and JavaScript.
// - Utility macros like `closure::Closure`: For handling JavaScript callbacks.
//
// This is essential for enabling the platform-specific logic to run in a browser
// or JavaScript environment when targeting WASM.
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// The `wgpu::InstanceDescriptor` is a configuration structure used to define the 
// parameters for creating a WGPU `Instance`. An `Instance` is the initialization 
// entry point for interacting with the GPU, and the descriptor allows customization 
// of certain features, such as enabling debugging utilities (e.g., `dx12` or `vulkan` 
// backends) or specifying backends explicitly (like `webgpu`, `vulkan`, `metal`, `dx12`, etc.).
use wgpu::InstanceDescriptor;

// The `std::sync::Arc` (Atomic Reference Counted) type is used to create a thread-safe, 
// reference-counted pointer. It enables multiple threads to share ownership of the same 
// data safely. Each clone of the `Arc` increments a reference count, and the data is 
// deallocated only when the last reference is dropped. This allows sharing immutable 
// data across threads without requiring locks.
use std::sync::Arc;

// The `web_time::{Duration, Instant}` module provides time handling functionality 
// compatible with both desktop and WebAssembly platforms. 
// - `Duration`: Represents a span of time, typically used for measuring time intervals.
// - `Instant`: Represents a specific point in time, useful for tracking elapsed durations 
//   (e.g., measuring the time between successive frame renders).
// This abstraction ensures consistent behavior across platforms with different time 
// handling APIs, such as std-based time on desktop and `performance.now()` on the web.
use web_time::{Duration, Instant};

// This import statement brings in essential components from the `winit` crate 
// to manage application handling, window events, and platform-specific interaction:
// - `ApplicationHandler`: A trait for handling the lifecycle of a winit application, 
//   including event processing and rendering.
// - `PhysicalSize`: Represents the physical dimensions of a window, 
//   typically used for handling resizing events or scaling.
// - `WindowEvent`: Enumerates various events that can occur on the window, 
//   such as input events, resizing, and focus changes.
// - `Theme`: Represents the current theme (light or dark) of the operating system, 
//   which can be used to adapt the application's appearance.
// - `Window`: Represents the application window, providing methods 
//   to control its properties, interact with the rendering surface, and handle events.
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    window::{Theme, Window},
};

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

/// The `Renderer` struct is responsible for rendering the application's graphical content,
/// including the 3D scene and GUI, using the `wgpu` and `egui_wgpu` frameworks.
///
/// It provides functionality to manage GPU resources, render pipelines, and textures
/// for both graphical and GUI elements. The `Renderer` struct integrates the GUI managed by 
/// `egui` with the 3D rendering provided by the application-specific scene.
///
/// # Fields
///
/// - `gpu`: A wrapper around WGPU-related resources, responsible for managing the `wgpu` device,
///   queue, and surface configuration.
/// - `depth_texture_view`: A depth texture view created for rendering 3D content. 
///   Uses `Depth32Float` format for depth calculations.
/// - `egui_renderer`: A renderer instance for rendering GUI elements created with `egui`.
/// - `scene`: The application's 3D scene, handling objects, transformations, and updates.
///
/// # Methods
///
/// - `new`: An asynchronous function that initializes the `Renderer` with the necessary GPU 
///   resources, depth texture, GUI renderer, and scene data.
/// - `resize`: Resizes the renderer's resources when the window or surface size changes.
/// - `render_frame`: Renders a frame by updating the scene, processing GUI commands, 
///   and combining all drawing operations into a final image for display.
///
/// # Usage
///
/// The `Renderer` is typically used within an application loop to handle real-time rendering
/// of both the GUI and the 3D graphics. It integrates `wgpu`'s rendering pipeline with `egui`'s 
/// GUI commands, ensuring seamless interaction between the two.
pub struct Renderer {
    /// A wrapper around WGPU-related resources, responsible for managing 
    /// the GPU's device, queue, and surface configuration.
    ///
    /// The `Gpu` struct handles the initialization and lifecycle of WGPU
    /// components such as the rendering device, command queue, and the surface
    /// which is used to present rendered frames. It also provides helper functions
    /// for creating textures and managing GPU-specific resources.
    gpu: Gpu,
    
    /// A depth texture view created for rendering 3D content.
    ///
    /// This texture view is used during the rendering process to store depth
    /// information for 3D scenes. It enables proper rendering of objects
    /// based on their relative depth to the camera, ensuring correct
    /// occlusion and object visibility.
    ///
    /// The depth texture is created with the `Depth32Float` format, which
    /// provides high precision for depth calculations. It is resized as
    /// needed when the window or surface size changes.
    depth_texture_view: wgpu::TextureView,
    
    /// A renderer instance for rendering GUI elements created with `egui`.
    ///
    /// This component is responsible for translating `egui`'s GUI 
    /// widgets and visuals into textures and commands that can be 
    /// rendered on the GPU using `wgpu`. It manages the lifecycle of 
    /// resources such as textures and buffers required for GUI rendering.
    ///
    /// The `egui_renderer` is tightly integrated with the rest of the 
    /// `Renderer`, ensuring seamless rendering alongside 3D scene 
    /// elements. It processes `egui` paint jobs and textures, and 
    /// combines them into the overall frame rendering pipeline.
    ///
    /// This field is initialized during the creation of the `Renderer` 
    /// struct and updated dynamically to reflect GUI state changes.
    egui_renderer: egui_wgpu::Renderer,
    
    /// The application's 3D scene, responsible for managing objects, transformations, 
    /// lighting, and other scene-specific logic.
    ///
    /// This field encapsulates the data and behavior needed to render and update the 3D scene. 
    /// It includes resources such as vertex and index buffers, shader pipelines, and 
    /// transformation matrices, ensuring that objects within the scene are rendered correctly.
    ///
    /// The `Scene` struct also handles per-frame updates, such as applying animations, 
    /// updating object positions, and recalculating camera matrices. It interacts closely with 
    /// the GPU to ensure efficient rendering.
    ///
    /// The scene is updated during the rendering process (`render_frame`), where it processes
    /// changes in the application's state or interactions initiated by the user.
    scene: Scene,
}

/// Implementation of the `Renderer` struct, which provides methods for managing 
/// and handling rendering tasks related to both the GUI and the 3D scene.
///
/// The `Renderer` struct integrates WGPU-based rendering for 3D graphics 
/// and the `egui` framework for GUI rendering, combining them into 
/// an efficient and reusable rendering pipeline.
///
/// # Constants
///
/// - `DEPTH_FORMAT`: Defines the texture format used for the depth buffer, 
///   which is critical for ensuring proper depth calculations in 3D rendering.
///
/// # Methods
///
/// - `new`: Asynchronous function to initialize the `Renderer` with specified 
///   GPU resources, depth textures, an `egui` renderer, and a 3D scene. 
///   This function prepares all essential components for rendering and is 
///   typically called during application startup.
///
/// - `resize`: Adjusts the size of GPU resources, such as the depth texture, 
///   to match the new dimensions of the window or surface. This is critical 
///   to maintaining proper rendering behavior when the window size changes.
///
/// - `render_frame`: Executes the rendering of a single frame, which includes:
///   - Updating the 3D scene based on the elapsed time.
///   - Processing `egui`-related GUI commands, such as texture updates.
///   - Issuing rendering commands to the GPU, combining GUI and 3D scene visuals, 
///     and presenting the final output to the surface.
///
/// The `Renderer` implementation is designed to be used in an application's main loop, 
/// updating and rendering the scene and GUI in real-time.
impl Renderer {
    /// The texture format used for the depth buffer in 3D rendering.
    ///
    /// This constant defines the format of the depth buffer as `Depth32Float`.
    /// It provides high precision for depth calculations, which is critical for
    /// rendering 3D scenes with accurate occlusion and depth-testing behavior.
    ///
    /// The depth buffer ensures that objects closer to the camera are drawn
    /// on top of those farther away, contributing to the realism of the scene.
    /// This format is particularly effective for applications that require precise
    /// depth calculations, such as rendering large, complex 3D environments.
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// Creates a new instance of the `Renderer` struct, initializing all required components.
    ///
    /// # Parameters
    ///
    /// - `window`: A target that represents the window or surface to which the application renders.
    /// - `width`: The width of the rendering surface in logical pixels.
    /// - `height`: The height of the rendering surface in logical pixels.
    ///
    /// # Returns
    ///
    /// A new `Renderer` instance with initialized GPU resources, depth textures, 
    /// an `egui` renderer, and a 3D scene. This setup provides the necessary components 
    /// for rendering both GUI elements and the 3D scene.
    ///
    /// # Asynchronous Behavior
    ///
    /// The function performs asynchronous operations to initialize GPU resources. 
    /// It is expected to be called during the application’s startup, often within an 
    /// asynchronous runtime or context.
    ///
    /// # Example
    ///
    /// ```rust
    /// let renderer = Renderer::new(window, 800, 600).await;
    /// ```
    ///
    /// In this example, a `Renderer` is initialized with a window surface and initial dimensions.
    ///
    /// # Panics
    ///
    /// This function may panic if GPU resources or the rendering environment cannot 
    /// be properly initialized.
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Self {
        // The GPU instance, responsible for managing the device, queue, and other
        // rendering-related resources required for interacting with the graphics hardware.
        //
        // This instance is initialized asynchronously and handles:
        // - Configuring the surface for rendering with appropriate settings.
        // - Managing the GPU device and command queue for issuing rendering commands.
        // - Creating and maintaining textures, buffers, and pipelines utilized during rendering.
        //
        // The `Gpu` struct is a key component for all rendering operations, abstracting
        // low-level GPU interactions and providing higher-level methods for resource creation
        // and management.
        let gpu = Gpu::new_async(window, width, height).await;
        
        // The texture view for the depth buffer used during 3D rendering.
        //
        // This texture view represents a depth buffer that is used to store depth
        // information for rendering operations. The depth buffer ensures correct
        // occlusion in 3D scenes, allowing closer objects to appear in front of those
        // farther away.
        //
        // The texture view is configured to match the dimensions of the rendering
        // surface and uses the depth format specified by `DEPTH_FORMAT`. It is
        // recreated whenever the rendering surface is resized to ensure accurate
        // depth calculations.
        //
        // This resource is critical for 3D rendering and is used as part of the
        // pipeline configuration.
        let depth_texture_view = gpu.create_depth_texture(width, height);

        // The `egui_renderer` is used to render the GUI elements within the application.
        //
        // This renderer integrates the `egui` framework with the GPU, enabling the
        // efficient rendering of GUI primitives on top of the 3D scene. It serves as
        // an important component of the rendering pipeline, ensuring that user-facing
        // interface elements are visually updated and properly displayed.
        //
        // # Details
        //
        // - Uses the `egui_wgpu` crate to bridge the `egui` GUI library and the `wgpu` graphics API.
        // - Configured with the device, surface format, and depth format to ensure compatibility
        //   with the GPU resources and rendering setup.
        // - Supports rendering of transparent and layered GUI elements, utilizing the depth buffer
        //   when required.
        //
        // This renderer is essential for applications with graphical interfaces, providing a bridge
        // between the interactive GUI and the underlying rendering engine.
        let egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            gpu.surface_config.format,
            Some(Self::DEPTH_FORMAT),
            1,
            false,
        );

        // The `scene` represents the 3D environment or visual content being rendered.
        //
        // This structure encapsulates all the objects, lighting, and other elements
        // that compose the 3D graphics rendered to the screen. It serves as the primary
        // container for managing and updating the visual state of the application.
        //
        // # Details
        //
        // - Configured using the GPU device and surface format to ensure compatibility
        //   with GPU resources and rendering workflows.
        // - Manages 3D models, textures, shaders, and other scene-related data.
        // - Integrates with the rendering pipeline, providing a primary source of data
        //   for rendering operations.
        //
        // The `scene` is updated and rendered as part of the rendering loop, reacting
        // to user input, animations, or external state to create an interactive and
        // dynamic 3D experience.
        let scene = Scene::new(&gpu.device, gpu.surface_format);

        Self {
            gpu,
            depth_texture_view,
            egui_renderer,
            scene,
        }
    }

    /// Resizes the rendering components to match the new size of the window or rendering surface.
    ///
    /// # Parameters
    ///
    /// - `width`: The new width of the rendering surface in logical pixels.
    /// - `height`: The new height of the rendering surface in logical pixels.
    ///
    /// # Details
    ///
    /// This method adjusts GPU resources, including the depth texture, to match
    /// the updated dimensions of the application window or surface. It ensures
    /// proper rendering by keeping the depth buffer and other components in sync
    /// with the current surface size.
    ///
    /// This function is typically called within the application when the user
    /// resizes the window, ensuring that the renderer properly updates to fit
    /// the new size.
    ///
    /// # Example
    ///
    /// ```rust
    /// renderer.resize(new_width, new_height);
    /// ```
    ///
    /// In this example, the renderer resizes its internal GPU resources to accommodate
    /// the updated surface dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
        self.depth_texture_view = self.gpu.create_depth_texture(width, height);
    }

    /// Renders a single frame, combining 3D scene rendering and `egui` GUI rendering.
    ///
    /// # Parameters
    ///
    /// - `screen_descriptor`: Describes the screen properties, such as size and DPI,
    ///   which are required by the `egui` renderer to render correctly.
    /// - `paint_jobs`: A collection of `egui` primitives (draw instructions)
    ///   to be rendered on the screen.
    /// - `textures_delta`: Contains information about added or removed textures
    ///   that are used by `egui` for rendering.
    /// - `delta_time`: The time elapsed since the last frame, which is used
    ///   for animations and time-dependent logic in the 3D scene.
    ///
    /// # Details
    ///
    /// This function coordinates the rendering of the GUI and the 3D scene. It updates
    /// the scene based on the elapsed time, updates `egui` textures, and manages the
    /// GPU command encoder for rendering to the surface. It performs the following steps:
    ///
    /// 1. Updates the 3D scene's state based on the `delta_time` to account for animations
    ///    or any other time-dependent behavior.
    /// 2. Updates and synchronizes `egui` textures (loading or freeing them as needed).
    /// 3. Executes `egui`'s paint jobs and integrates them with the rendered output.
    /// 4. Renders the frame to the surface, using a render pass that includes a color attachment
    ///    for the main scene and a depth attachment for proper depth-testing.
    ///
    /// # Panics
    ///
    /// This function will panic if the process of retrieving the surface texture fails. This
    /// can occur if the surface becomes invalid (e.g., if the window is resized or destroyed).
    ///
    /// # Example
    ///
    /// ```rust
    /// renderer.render_frame(
    ///     screen_descriptor,
    ///     paint_jobs,
    ///     textures_delta,
    ///     delta_time,
    /// );
    /// ```
    ///
    /// In this example, the function is called to render one frame, using the provided
    /// `egui` render data, screen dimensions, and the time elapsed since the previous frame.
    pub fn render_frame(
        &mut self,
        screen_descriptor: egui_wgpu::ScreenDescriptor,
        paint_jobs: Vec<egui::epaint::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        delta_time: crate::Duration,
    ) {
        // The elapsed time since the last frame, in seconds, represented as a 32-bit floating-point number.
        // This variable is used to update the state of the 3D scene, animations, and other
        // time-dependent logic within the render pipeline.
        //
        // `delta_time` facilitates smooth animations and transitions by allowing computations
        // to take the elapsed time into account, ensuring consistent behavior regardless of frame rate.
        let delta_time = delta_time.as_secs_f32();

        self.scene
            .update(&self.gpu.queue, self.gpu.aspect_ratio(), delta_time);

        // This loop iterates over all texture changes in the `textures_delta.set` map,
        // where `id` is the unique identifier for a texture and `image_delta` describes
        // the changes to be applied to that texture. For each entry, it updates
        // the corresponding texture in the `egui_renderer` using the provided
        // `gpu.device` and `gpu.queue`. This ensures that `egui` textures are
        // synchronized with changes made to them.
        for (id, image_delta) in &textures_delta.set {
            self.egui_renderer
                .update_texture(&self.gpu.device, &self.gpu.queue, *id, image_delta);
        }

        // Iterate through all texture updates specified in the `textures_delta.set` map.
        // Each entry in the map consists of a texture `id` (a unique identifier) and
        // an `image_delta` (describing how the texture should be updated). The `egui_renderer`
        // is then instructed to update these textures using the provided `gpu.device` and
        // `gpu.queue` for GPU operations, ensuring `egui` texture data is synchronized with
        // the latest changes.
        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // A command encoder used to record a series of GPU commands for execution.
        //
        // The `encoder` is responsible for managing the commands that will be sent to the GPU.
        // It is used to record operations such as rendering commands, buffer updates, and texture
        // manipulations. After recording, the commands are submitted for execution on the GPU queue.
        //
        // # Details
        //
        // - The encoder is created using the `wgpu::Device` method `create_command_encoder`.
        // - It has a descriptive label associated with it (`"Render Encoder"`) for easier debugging and profiling.
        // - This encoder is used to manage commands for updating `egui` buffers, rendering the 3D scene,
        //   updating textures, and executing draw calls on the primary surface texture.
        //
        // At the end of the rendering process, the recorded commands in the encoder are finalized by
        // calling the `.finish()` method and submitted to the GPU queue for execution.
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // Represents the texture for the current frame, retrieved from the surface.
        //
        // The `surface_texture` is obtained using the `get_current_texture()` method on the GPU surface.
        // It represents the texture that will be used as the rendering target for the current frame.
        // Rendering commands are issued to draw content onto this texture.
        //
        // # Details
        //
        // - This texture is tied to the window or canvas the application is rendering to.
        // - The texture is presented using the `present()` method once rendering has finished.
        // - If acquiring the texture fails (e.g., due to a lost surface), an error will be raised.
        //
        // The `surface_texture` is vital for ensuring rendered frames are output to the display.
        let surface_texture = self
            .gpu
            .surface
            .get_current_texture()
            .expect("Failed to get surface texture!");

        // Represents a view of the texture for the current frame.
        //
        // The `surface_texture_view` is created from the `surface_texture` using the `create_view` method.
        // It defines how the texture resource should be accessed and what properties it should expose
        // for rendering operations. This is especially useful when multiple views of the same texture
        // are required with varying configurations.
        //
        // # Details
        //
        // - The view is created with the `wgpu::TextureViewDescriptor`, which specifies configuration
        //   options such as format, dimension, aspect, and mip levels.
        // - The `aspect` defaults to full visibility (color only for color textures).
        // - The view is required as input for render operations like `RenderPass` or `RenderBundle`.
        // - This view allows the GPU to interpret the `surface_texture` data correctly when rendering
        //   the frame onto the screen.
        let surface_texture_view =
            surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: wgpu::Label::default(),
                    aspect: wgpu::TextureAspect::default(),
                    format: Some(self.gpu.surface_format),
                    dimension: None,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                    usage: None,
                });

        encoder.insert_debug_marker("Render scene");

        // This scope around the crate::render_pass prevents the
        // crate::render_pass from holding a borrow to the encoder,
        // which would prevent calling `.finish()` in
        // preparation for queue submission.
        {
            // Represents the render pass for issuing rendering commands to the GPU.
            //
            // The `render_pass` variable is used to manage and record a sequence of rendering operations,
            // such as setting pipeline states, drawing, and clearing attachments. This ensures the GPU
            // can execute these operations efficiently.
            //
            // # Details
            //
            // - A `render_pass` is created using the `encoder.begin_render_pass()` method with a
            //   `wgpu::RenderPassDescriptor` that specifies configurations for color and depth attachments.
            // - The `color_attachments` field associates the render target texture (or view) with the
            //   operations to be performed (e.g., clearing or storing color data).
            // - The `depth_stencil_attachment` field manages the depth and stencil buffers, ensuring proper
            //   depth testing and rendering order.
            //
            // This render pass is scoped to ensure the `'encode` lifetime of the encoder is not held after
            // the render pass completes, allowing the `encoder` to be finalized with the `.finish()` method.
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.19,
                            g: 0.24,
                            b: 0.42,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.scene.render(&mut render_pass);

            self.egui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }
}

/// A struct representing the GPU-related resources and configurations required for rendering.
///
/// This struct manages the GPU surface, device, queue, surface configuration, and provides utility methods
/// for resizing, creating textures, and handling aspect ratios.
///
/// # Fields
/// - `surface` (`wgpu::Surface`): Represents the surface associated with the GPU rendering target, typically
///   tied to a window or canvas.
/// - `device` (`wgpu::Device`): The device instance that is used to create GPU-dependent resources and execute commands.
/// - `queue` (`wgpu::Queue`): The command queue, used to submit command buffers to the GPU for execution.
/// - `surface_config` (`wgpu::SurfaceConfiguration`): The configuration settings for the rendering surface,
///   such as its size, format, and other parameters.
/// - `surface_format` (`wgpu::TextureFormat`): The texture format used by the surface, obtained from the surface's capabilities.
///
/// # Methods
/// - `aspect_ratio() -> f32`: Computes the aspect ratio of the rendering surface based on the current width and height.
/// - `resize(width: u32, height: u32)`: Resizes the rendering surface to the specified dimensions and updates its configuration.
/// - `create_depth_texture(width: u32, height: u32) -> wgpu::TextureView`: Creates and returns a depth texture
///   for use in rendering, based on the specified dimensions.
/// - `new_async(window, width, height) -> Self`: Asynchronously initializes a `Gpu` instance with the specified
///   window and dimensions.
///
/// # Example
/// ```rust
/// use wgpu::Surface;
///
/// // Create a Gpu instance using a window reference and dimensions.
/// async fn create_gpu(window: winit::window::Window, width: u32, height: u32) -> Gpu {
///     Gpu::new_async(window, width, height).await
/// }
/// ```
pub struct Gpu {
    /// The surface associated with the GPU rendering target.
    ///
    /// This represents the platform-specific surface on which rendering operations 
    /// are performed. Typically, it is tied to a window or a canvas, allowing the
    /// rendered output to be displayed to the user.
    ///
    /// The surface is used to configure the swap chain and ensures that rendered
    /// frames are presented on the screen.
    pub surface: wgpu::Surface<'static>,
    
    /// The device instance that is used to create GPU-dependent resources and execute commands.
    ///
    /// This is a handle to the graphics device provided by the system's GPU.
    /// It is responsible for resource creation (e.g., buffers, textures, and pipelines)
    /// as well as issuing command submissions to the GPU queue for execution.
    ///
    /// The `device` is obtained from a suitable adapter during initialization and
    /// exposed here to allow interaction with the GPU for rendering and computation tasks.
    /// It is essential for managing the lifetime and scope of GPU resources.
    pub device: wgpu::Device,
    
    /// The command queue, used to submit command buffers to the GPU for execution.
    ///
    /// This queue handles the actual execution of GPU commands by dispatching
    /// command buffers to the GPU for processing. It serves as a bridge between
    /// CPU-side operations and the GPU.
    ///
    /// It is a critical part of the rendering pipeline, enabling developers to perform
    /// draw calls, resource uploads, and synchronization tasks necessary for
    /// rendering and computations.
    ///
    /// The queue is created alongside the device and used throughout the lifecycle
    /// of rendering to ensure commands are executed efficiently.
    pub queue: wgpu::Queue,
    
    /// The configuration settings for the rendering surface.
    ///
    /// This field holds an instance of `wgpu::SurfaceConfiguration`, which defines
    /// how the GPU surface is configured for rendering. Configuration includes
    /// parameters such as the width, height, texture format, and other pertinent
    /// details required for proper operation of the surface.
    ///
    /// The surface configuration is essential for ensuring compatibility between
    /// the rendering surface and the device. It is used for resizing the surface
    /// and reconfiguring it when the window is resized or other configuration
    /// changes are necessary.
    ///
    /// Typical usage involves providing this configuration during surface creation
    /// or dynamically updating it during runtime to accommodate varying system
    /// conditions or user interactions.
    pub surface_config: wgpu::SurfaceConfiguration,
    
    /// The texture format of the rendering surface.
    ///
    /// This field specifies the texture format (`wgpu::TextureFormat`) used for the surface,
    /// which determines the color format and data layout of the rendered output.
    ///
    /// The texture format is typically selected based on the surface's capabilities
    /// and represents the format in which rendered frames are stored before being
    /// presented to the screen.
    ///
    /// It is used when configuring the surface and is essential for ensuring compatibility
    /// between the surface and the rendering pipeline.
    ///
    /// Common formats include `Bgra8Unorm` and `Rgba8Unorm`, depending on the system and
    /// rendering requirements.
    pub surface_format: wgpu::TextureFormat,
}

/// Implementation block for the `Gpu` struct, providing utility functions
/// to work with the GPU, such as querying the display's aspect ratio,
/// resizing the rendering surface, and creating GPU-dependent resources.
impl Gpu {
    /// Calculates the aspect ratio of the rendering surface.
    ///
    /// The aspect ratio is determined by dividing the width of the surface by its height.
    /// This value is useful for correctly scaling rendered content to fit the target display.
    ///
    /// # Returns
    ///
    /// A `f32` value representing the aspect ratio of the surface.
    /// If the height of the surface is zero, the function ensures no division by zero
    /// by using `1` as the minimum value for height in the calculation.
    ///
    /// # Examples
    ///
    /// Assuming the surface width is 1920 and height is 1080:
    ///
    /// ```
    /// let aspect_ratio = gpu.aspect_ratio();
    /// assert_eq!(aspect_ratio, 1920.0 / 1080.0);
    /// ```
    pub fn aspect_ratio(&self) -> f32 {
        self.surface_config.width as f32 / self.surface_config.height.max(1) as f32
    }

    /// Resizes the rendering surface to the specified dimensions.
    ///
    /// This method updates the `surface_config` of the GPU to match the new width
    /// and height values and reconfigures the rendering surface accordingly.
    ///
    /// It is typically used when the window is resized or when the rendering surface
    /// requires reconfiguration during runtime.
    ///
    /// # Parameters
    ///
    /// - `width`: The new width of the rendering surface, in pixels.
    /// - `height`: The new height of the rendering surface, in pixels.
    ///
    /// # Example
    ///
    /// Assume the rendering surface needs to be resized to 1280x720:
    ///
    /// ```
    /// gpu.resize(1280, 720);
    /// ```
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    /// Creates a depth texture for the GPU rendering pipeline.
    ///
    /// The depth texture is used for managing depth testing during rendering,
    /// ensuring that objects closer to the camera obscure ones further away.
    ///
    /// This method creates a `wgpu::Texture` configured for depth rendering,
    /// and returns its corresponding `wgpu::TextureView` for further use.
    ///
    /// # Parameters
    ///
    /// - `width`: The width of the depth texture, in pixels.
    /// - `height`: The height of the depth texture, in pixels.
    ///
    /// # Returns
    ///
    /// A `wgpu::TextureView` representing the depth texture view, which can
    /// be used as the depth attachment in a render pass.
    ///
    /// # Remarks
    ///
    /// - The depth texture uses the `Depth32Float` format, which stores a 32-bit
    ///   floating-point value for depth information.
    /// - It supports `RENDER_ATTACHMENT` and `TEXTURE_BINDING` usages, making it
    ///   suitable for rendering and sampling.
    ///
    /// # Examples
    ///
    /// ```
    /// let depth_texture = gpu.create_depth_texture(1920, 1080);
    /// ```
    pub fn create_depth_texture(&self, width: u32, height: u32) -> wgpu::TextureView {
        // The `texture` variable represents the GPU resource for the depth texture.
        //
        // It is created using the `create_texture` method, which defines the texture's
        // properties such as its dimensions, format, and usage. In this context, the
        // texture is specifically configured for depth rendering.
        //
        // # Properties
        //
        // - **Label**: The texture is labeled as "Depth Texture" for debugging purposes.
        // - **Size**: The texture has a width and height defined by the method parameters,
        //   with a depth of 1 since it is a 2D texture.
        // - **Format**: The format is set to `Depth32Float`, which provides high precision
        //   for depth calculations.
        // - **Usage**: The texture supports `RENDER_ATTACHMENT` (used in the rendering pipeline)
        //   and `TEXTURE_BINDING` (allowing it to be sampled if necessary).
        //
        // # Remarks
        //
        // The `texture` variable is central to enabling proper depth testing during rendering,
        // ensuring the visual correctness of overlapping objects in the scene.
        let texture = self.device.create_texture(
            &(wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }),
        );
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            base_array_layer: 0,
            array_layer_count: None,
            mip_level_count: None,
            usage: None,
        })
    }

    /// Creates a new GPU context asynchronously.
    ///
    /// This method initializes the GPU context using the given window and dimensions.
    /// It configures the surface, requests an adapter, and sets up a compatible device and queue.
    ///
    /// This function is `async` because it relies on asynchronous GPU resource requests.
    ///
    /// # Parameters
    ///
    /// - `window`: The target window for rendering. This is used to create the GPU surface.
    /// - `width`: The initial width of the rendering surface, in pixels.
    /// - `height`: The initial height of the rendering surface, in pixels.
    ///
    /// # Returns
    ///
    /// Returns an instance of `Self` containing the configured GPU context, including
    /// the `surface`, `device`, `queue`, and `surface_config`.
    ///
    /// # Remarks
    ///
    /// - The method selects an adapter that is compatible with the provided window surface.
    /// - It ensures the surface configuration matches the surface's capabilities, selecting
    ///   a non-sRGB format for compatibility with `egui`.
    /// - Errors are logged via the `log` crate if any GPU device or adapter setup fails.
    ///
    /// # Errors
    ///
    /// Panics if it fails to create a surface, request an adapter, or request a device.
    ///
    /// # Examples
    ///
    /// ```
    /// let gpu_context = GpuContext::new_async(window, 1920, 1080).await;
    /// ```
    pub async fn new_async(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Self {
        // The `instance` variable represents a handle to the WGPU instance,
        // which is the entry point for interacting with the GPU.
        //
        // # Remarks
        //
        // - The `Instance` is used to create GPU surfaces and query available adapters.
        // - It serves as the foundational object for setting up GPU-related resources.
        // - A single `Instance` can manage multiple surfaces and adapters.
        let instance = wgpu::Instance::new(&InstanceDescriptor::default());
        
        // The `surface` variable represents the rendering surface associated with the given window.
        //
        // # Remarks
        //
        // - The surface is created using the `Instance` and is tied to the windowing system.
        // - It serves as the GPU's target for presenting rendered frames.
        // - This surface will be configured later with a suitable format and dimensions
        //   based on the adapter’s capabilities.
        //
        // - The `Surface` is essential for rendering in windowed applications, as it allows
        //   the GPU to directly render content to the display.
        let surface = instance.create_surface(window).unwrap();

        // Represents a handle to the GPU adapter used for device creation.
        //
        // # Remarks
        //
        // - The `adapter` is selected based on the provided options, such as power preference
        //   and compatibility with the rendering surface.
        // - The `adapter` provides information about the GPU and its capabilities, such as
        //   supported features, limits, and formats.
        // - It plays a central role in determining whether the system's GPU meets the
        //   requirements for rendering and computation tasks.
        //
        // # Error Handling
        //
        // - If no suitable adapter is found that matches the provided options, the operation
        //   will fail, causing the application to panic with an appropriate error message.
        //
        // # Example Usage
        //
        // ```rust
        // let adapter = instance
        //     .request_adapter(&wgpu::RequestAdapterOptions {
        //         power_preference: wgpu::PowerPreference::HighPerformance,
        //         compatible_surface: Some(&surface),
        //         force_fallback_adapter: false,
        //     })
        //     .await
        //     .expect("Failed to request adapter!");
        // ```
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to request adapter!");

        // Represents the GPU device used for rendering and computation.
        //
        // # Remarks
        //
        // - The `device` is created from the selected GPU adapter and serves as the primary
        //   interface for interacting with the GPU.
        // - It allows the creation of various resources such as buffers, textures, and pipelines,
        //   which are essential for rendering and compute tasks.
        // - The `device` also manages the underlying GPU's state and ensures efficient resource usage.
        //
        // # Importance
        //
        // - The `device` is a central component in the WGPU pipeline, enabling the execution of
        //   GPU-based operations.
        // - With the `queue`, it forms the foundation for submitting commands to the GPU.
        //
        // # Error Handling
        //
        // - If the GPU device fails to be created due to invalid configurations or hardware issues,
        //   the operation will fail, and the application will panic with an error message.
        //
        // ---
        //
        // Represents the command queue associated with the GPU device.
        //
        // # Remarks
        //
        // - The `queue` is responsible for submitting command buffers to the GPU for execution.
        // - It provides an interface for queuing up and executing GPU commands related to rendering and computation.
        // - The `queue` works in conjunction with the `device` to facilitate the rendering pipeline.
        //
        // # Importance
        //
        // - The `queue` plays a critical role in orchestrating the GPU's workflow by scheduling and
        //   managing the execution of tasks.
        // - It allows the application to send rendering commands or any GPU workload efficiently.
        //
        // # Error Handling
        //
        // - If the `queue` fails to process commands due to invalid configurations or resource limitations,
        //   it may lead to rendering errors or interruptions in the application's graphical output.
        let (device, queue) = {
            log::info!("WGPU Adapter Features: {:#?}", adapter.features());
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("WGPU Device"),
                        memory_hints: wgpu::MemoryHints::default(),
                        required_features: wgpu::Features::default(),
                        #[cfg(not(target_arch = "wasm32"))]
                        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                        #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
                        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                        #[cfg(all(target_arch = "wasm32", feature = "webgl"))]
                        required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                            .using_resolution(adapter.limits()),
                    },
                    None,
                )
                .await
                .expect("Failed to request a device!")
        };

        // Represents the capabilities of the surface as determined by the selected GPU adapter.
        //
        // # Remarks
        //
        // - The `surface_capabilities` contain information about what surface configurations are supported
        //   by the GPU and the display system.
        // - These capabilities include supported texture formats, presentation modes, and alpha compositing modes.
        //
        // # Fields
        //
        // - `formats`: A list of `wgpu::TextureFormat` specifying which texture formats are supported by the surface.
        // - `present_modes`: A list of `wgpu::PresentMode` enumerating the possible methods of presenting the surface's textures
        //   (e.g., V-Sync, Mailbox, Immediate, etc.).
        // - `alpha_modes`: A list of `wgpu::CompositeAlphaMode` that describe how the alpha channel is handled during compositing.
        //
        // # Importance
        //
        // - Knowing the surface capabilities is essential for configuring the surface correctly to match
        //   the desired performance, visual quality, and compatibility requirements.
        // - This ensures that the selected `format`, `present_mode`, and `alpha_mode` align with what the
        //   GPU and platform can support.
        //
        // # Usage Example
        //
        // ```rust
        // let surface_format = surface_capabilities
        //     .formats
        //     .iter()
        //     .copied()
        //     .find(|f| !f.is_srgb())
        //     .unwrap_or(surface_capabilities.formats[0]);
        // ```
        let surface_capabilities = surface.get_capabilities(&adapter);

        // The surface texture format selected for rendering.
        //
        // # Remarks
        //
        // - The `surface_format` specifies the format of the texture that will be used for rendering operations.
        // - It is chosen based on the capabilities of the surface, ensuring compatibility with the GPU and the rendering pipeline.
        // - A non-sRGB format is preferred by default as it aligns with the requirements of the `egui` library.
        //
        // # Purpose
        //
        // - Defines the format of the surface textures, which includes properties like color depth, color space, and compression.
        // - Ensures that the texture format matches both the application's needs and the GPU's capabilities.
        //
        // # Importance
        //
        // - The texture format impacts the quality and performance of the rendering.
        // - By selecting an appropriate format, the rendering pipeline can achieve optimal color reproduction and efficiency.
        //
        // # Error Handling
        //
        // - If no suitable format is found, it defaults to the first format in the list supported by the surface capabilities.
        //
        // # Usage Example
        //
        // ```rust
        // let surface_format = surface_capabilities
        //     .formats
        //     .iter()
        //     .copied()
        //     .find(|f| !f.is_srgb()) // egui prefers non-sRGB surface texture
        //     .unwrap_or(surface_capabilities.formats[0]);
        // ```
        let surface_format = surface_capabilities
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb()) // egui wants a non-srgb surface texture
            .unwrap_or(surface_capabilities.formats[0]);

        // The configuration settings for the surface.
        //
        // # Remarks
        //
        // - The `surface_config` defines how the surface should operate and interact with the
        //   GPU, including details like texture usage, format, dimensions, and presentation mode.
        // - Properly configuring the surface ensures that rendering can proceed efficiently
        //   and with the desired visual quality.
        //
        // # Fields
        //
        // - `usage`: Specifies how the texture will be used. The `RENDER_ATTACHMENT` usage ensures
        //   that it will be utilized for rendering operations.
        // - `format`: The selected texture format (`surface_format`) for the surface.
        // - `width`: Width of the surface in pixels.
        // - `height`: Height of the surface in pixels.
        // - `present_mode`: Determines how the surface's textures are presented. This impacts
        //   performance and visual behavior (e.g., vsync).
        // - `alpha_mode`: Specifies how the alpha channel is handled during compositing.
        // - `view_formats`: A list of additional view formats supported by the surface textures.
        // - `desired_maximum_frame_latency`: Controls the number of frames the GPU can process
        //   ahead, which affects latency and throughput.
        //
        // # Importance
        //
        // - Correct configuration is crucial for compatibility with the display system and
        //   ensuring smooth and artifact-free rendering.
        // - It allows the application to match the requirements of specific rendering pipelines
        //   and desired user experience (e.g., reduced latency or increased frame rates).
        //
        // # Usage Example
        //
        // ```rust
        // let surface_config = wgpu::SurfaceConfiguration {
        //     usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        //     format: surface_format,
        //     width,
        //     height,
        //     present_mode: surface_capabilities.present_modes[0],
        //     alpha_mode: surface_capabilities.alpha_modes[0],
        //     view_formats: vec![],
        //     desired_maximum_frame_latency: 2,
        // };
        // ```
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_capabilities.present_modes[0],
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        Self {
            surface,
            device,
            queue,
            surface_config,
            surface_format,
        }
    }
}

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
struct Scene {
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

/// Represents a single vertex in a 3D scene, including its position and color attributes.
///
/// This struct is used to define the data structure for vertices passed to the GPU 
/// through vertex buffers. Each `Vertex` object contains a 4D position vector and a 4D 
/// color vector, both stored as arrays of `f32`.
///
/// # Fields
///
/// - `position`: A `[f32; 4]` array representing the position of the vertex in 3D space.
///   The fourth component is typically used for homogenous coordinates in rendering pipelines.
/// - `color`: A `[f32; 4]` array representing the color of the vertex. The values are
///   typically normalized between 0.0 and 1.0, corresponding to RGBA components.
///
/// # Usage
///
/// This structure, when combined with the `Vertex` trait, assists in creating 
/// vertex buffer layouts for passing data to a graphics pipeline.
///
/// ```rust
/// let vertex = Vertex {
///     position: [0.0, 1.0, 0.0, 1.0],
///     color: [1.0, 0.0, 0.0, 1.0],
/// };
/// ```
///
/// Additionally, the `vertex_attributes` and `description` methods can be used to
/// define the vertex buffer layout for the GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    /// The position of the vertex in 3D space.
    ///
    /// This is a `[f32; 4]` array, where the first three elements represent the 
    /// x, y, and z coordinates, and the fourth element is typically used for 
    /// homogenous coordinates in 3D graphics pipelines. This allows for operations 
    /// such as perspective division during rendering.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vertex_position = [0.0, 1.0, 0.0, 1.0];
    /// ```
    position: [f32; 4],
    
    /// The color of the vertex.
    ///
    /// This is a `[f32; 4]` array, representing the RGBA (Red, Green, Blue, Alpha) components 
    /// of the vertex's color. Each component is typically normalized between 0.0 and 1.0, 
    /// where 0.0 represents no intensity and 1.0 represents full intensity.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vertex_color = [1.0, 0.0, 0.0, 1.0]; // Red color with full opacity
    /// ```
    color: [f32; 4],
}

/// Implementation of methods for the `Vertex` struct which represents a 3D model vertex.
///
/// The `vertex_attributes` and `description` methods define how the vertex data
/// is laid out in memory for the GPU. This includes details on position and color
/// attributes supported by the vertex shader.
///
/// # Methods
///
/// - `vertex_attributes`: Generates a list of vertex attributes for this struct.
///     This defines which attributes the GPU expects and their data types.
///
/// - `description`: Returns the vertex buffer layout, which describes the memory
///     layout of the vertex buffer. It includes the stride, step mode, and 
///     attributes.
///
/// # Usage
///
/// These methods are essential for preparing data to be sent to the graphics API.
///
/// ```rust
/// let attributes = Vertex::vertex_attributes();
/// let layout = Vertex::description(&attributes);
/// ```
///
/// The layout is passed during pipeline creation, while the attributes are
/// used to create shaders and bind proper data from the buffer.
impl Vertex {
    /// Generates the vertex attributes layout for the `Vertex` struct.
    ///
    /// This method defines how the vertex data is interpreted by the GPU, specifying
    /// the attributes for the `position` and `color` fields in the vertex shader.
    ///
    /// # Returns
    ///
    /// A `Vec<wgpu::VertexAttribute>` array consisting of two vertex attributes:
    ///
    /// - The first attribute corresponds to the `position` field and is represented 
    ///   as a 4-component floating-point vector (`Float32x4`).
    /// - The second attribute corresponds to the `color` field and is represented 
    ///   as a 4-component floating-point vector (`Float32x4`).
    ///
    /// These attributes are indexed starting from 0 in the vertex shader.
    ///
    /// # Example
    ///
    /// ```rust
    /// let attributes = Vertex::vertex_attributes();
    /// // attributes[0] will represent the layout for `position`
    /// // attributes[1] will represent the layout for `color`
    /// ```
    ///
    /// # GPU Compatibility
    ///
    /// This layout is essential for configuring how the GPU interprets vertex data 
    /// passed to it during rendering pipeline setup.
    pub fn vertex_attributes() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4].to_vec()
    }

    /// Returns the vertex buffer layout for the `Vertex` struct.
    ///
    /// This method defines the memory layout of vertex data for the GPU, including 
    /// the stride (size of a single vertex in bytes), vertex step mode, and the attributes 
    /// (e.g., position and color). The vertex buffer layout is used during pipeline creation 
    /// to correctly interpret and bind vertex data.
    ///
    /// # Parameters
    ///
    /// - `attributes`: A reference to a slice of `wgpu::VertexAttribute`, which defines 
    ///   the vertex attributes (such as position and color) and their data types.
    ///
    /// # Returns
    ///
    /// A `wgpu::VertexBufferLayout` struct containing:
    ///
    /// - `array_stride`: Size of a single vertex, calculated as the size of the `Vertex` struct.
    /// - `step_mode`: Defines how vertices are processed (e.g., per-vertex or per-instance). 
    ///   This defaults to `wgpu::VertexStepMode::Vertex`, which processes one vertex at a time.
    /// - `attributes`: The provided slice of vertex attributes, describing how the vertex data 
    ///   is laid out in memory.
    ///
    /// # Example
    ///
    /// ```rust
    /// let attributes = Vertex::vertex_attributes();
    /// let layout = Vertex::description(&attributes);
    /// // Use `layout` for configuring the vertex buffer in the GPU pipeline.
    /// ```
    ///
    /// # GPU Compatibility
    ///
    /// The returned layout is critical for setting up the rendering pipeline, ensuring that
    /// vertex data is interpreted correctly according to the defined attributes and their memory layout.
    pub fn description(attributes: &[wgpu::VertexAttribute]) -> wgpu::VertexBufferLayout {
        // Constructs a `wgpu::VertexBufferLayout` using the provided vertex attributes, defining the memory 
        // layout for a single vertex, including its stride, step mode, and attributes. This layout is used to 
        // inform the GPU how vertex data is structured and accessed during rendering.
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes,
        }
    }
}

/// Represents the uniform buffer used to pass data from the CPU to the GPU.
///
/// This struct contains a single field, `mvp`, which is a 4x4 matrix used for
/// Model-View-Projection (MVP) transformations in the rendering pipeline.
/// These transformations are typically applied to vertices during rendering
/// operations to ensure proper positioning, perspective, and scaling of 
/// rendered objects.
///
/// # Fields
///
/// - `mvp`: A 4x4 matrix (`nalgebra_glm::Mat4`) used for MVP transformations.
///
/// # Memory Layout
///
/// The struct is annotated with `#[repr(C)]` and traits from the `bytemuck` crate,
/// ensuring it is compatible with GPU memory layouts. This enables it to be
/// safely and directly transferred to GPU buffer memory.
///
/// - `#[repr(C)]`: Ensures the struct uses a C-compatible memory layout.
/// - `bytemuck::Pod` and `bytemuck::Zeroable`: Allow conversion between the
///   struct and a binary slice representation, which is essential for loading
///   the data into a `wgpu` buffer.
///
/// # Example
///
/// Using this struct with a uniform buffer:
///
/// ```rust
/// let uniform_instance = UniformBuffer {
///     mvp: nalgebra_glm::identity(),
/// };
/// // Pass this buffer to the GPU via `wgpu::Buffer`
/// ```
#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformBuffer {
    /// A 4x4 matrix (`nalgebra_glm::Mat4`) representing the Model-View-Projection (MVP) transformations.
    ///
    /// This matrix is used by the GPU rendering pipeline to transform vertex positions
    /// from model space to screen space. The MVP matrix consists of the following components:
    ///
    /// - **Model transformation**: Positions objects in the world, affecting their scale,
    ///   rotation, and translation.
    /// - **View transformation**: Handles the camera's position and orientation, modifying
    ///   how the scene is viewed.
    /// - **Projection transformation**: Applies a perspective or orthographic projection
    ///   to account for screen-space mapping and depth.
    ///
    /// This property is designed to be compatible with GPU memory and is efficiently
    /// transferred to the GPU uniform buffer for real-time rendering.
    mvp: nalgebra_glm::Mat4,
}

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
struct UniformBinding {
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

/// An array of `Vertex` instances representing the vertices of a triangle.
///
/// Each vertex includes:
/// - `position`: The 4D coordinates of the vertex in homogeneous space, defined as `[x, y, z, w]`.
/// - `color`: The color of the vertex represented as `[r, g, b, a]` with channels specified in the range `[0.0, 1.0]`.
///
/// This triangle is defined in a right-handed coordinate system:
/// - The first vertex is located at `[1.0, -1.0, 0.0, 1.0]` with red color `[1.0, 0.0, 0.0, 1.0]`.
/// - The second vertex is at `[-1.0, -1.0, 0.0, 1.0]` with green color `[0.0, 1.0, 0.0, 1.0]`.
/// - The third vertex is at `[0.0, 1.0, 0.0, 1.0]` with blue color `[0.0, 0.0, 1.0, 1.0]`.
///
/// These vertices can be used to construct a basic triangle in a graphics pipeline, where each vertex
/// contributes to the shape and color of the rendered primitive.
const VERTICES: [Vertex; 3] = [
    Vertex {
        position: [1.0, -1.0, 0.0, 1.0],
        color: [1.0, 0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0, 1.0],
        color: [0.0, 1.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.0, 1.0, 0.0, 1.0],
        color: [0.0, 0.0, 1.0, 1.0],
    },
];

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
const INDICES: [u32; 3] = [0, 1, 2]; // Clockwise winding order

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
const SHADER_SOURCE: &str = "
struct Uniform {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> ubo: Uniform;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = vert.color;
    out.position = ubo.mvp * vert.position;
    return out;
};

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color);
}
";
