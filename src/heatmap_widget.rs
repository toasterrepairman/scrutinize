use gtk::prelude::*;
use gtk::{DrawingArea, Box as GtkBox, Orientation, Scale};
use std::cell::RefCell;
use std::rc::Rc;
use crate::tensor_slice::SliceSelection;

/// Configuration for heatmap rendering
const MAX_HEATMAP_DIMENSION: usize = 2000; // Max resolution for heatmap visualization (2K x 2K)
const MIN_CELL_SIZE: f64 = 1.0; // Minimum pixel size for each cell (prevents too small cells)
const MAX_DISPLAY_WIDTH: f64 = 600.0; // Maximum widget width (reasonable size for UI)
const MAX_DISPLAY_HEIGHT: f64 = 600.0; // Maximum widget height (reasonable size for UI)
const VIEWPORT_WIDTH: f64 = 500.0; // Default viewport width for large tensors
const VIEWPORT_HEIGHT: f64 = 400.0; // Default viewport height for large tensors

#[derive(Debug, Clone)]
pub struct TensorStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub sparsity: f32, // Percentage of near-zero values
    pub median: f32,
}

#[derive(Clone)]
pub struct HeatmapWidget {
    container: GtkBox,  // Main container (includes slice controls + drawing area)
    drawing_area: DrawingArea,
    data: Rc<RefCell<Option<HeatmapData>>>,
    display_mode: Rc<RefCell<DisplayMode>>,
    tooltip_label: gtk::Label,
    slice_selection: Rc<RefCell<Option<SliceSelection>>>,
    slice_controls: Rc<RefCell<Option<GtkBox>>>,  // Container for slice controls
    on_slice_change: Rc<RefCell<Option<Box<dyn Fn(SliceSelection)>>>>,  // Callback when slice changes
    active_selection: Rc<RefCell<Option<SliceSelection>>>,  // Keep the active selection alive for slider callbacks
}

#[derive(Clone, Copy, PartialEq)]
enum DisplayMode {
    Heatmap,
    Histogram,
    LinePlot,
}

#[derive(Clone)]
struct HeatmapData {
    values: Vec<f32>,
    width: usize,
    height: usize,
    min_value: f32,
    max_value: f32,
}

impl HeatmapWidget {
    pub fn new() -> Self {
        // Create main container
        let container = GtkBox::new(Orientation::Vertical, 8);

        let drawing_area = DrawingArea::new();

        // Set strict size constraints to prevent layout crashes
        drawing_area.set_content_width(
            (VIEWPORT_WIDTH as i32).min(500)
        );
        drawing_area.set_content_height(
            (VIEWPORT_HEIGHT as i32).min(400)
        );
        drawing_area.set_size_request(
            (VIEWPORT_WIDTH as i32).min(600),
            (VIEWPORT_HEIGHT as i32).min(500)
        );

        let data = Rc::new(RefCell::new(None));
        let display_mode = Rc::new(RefCell::new(DisplayMode::Heatmap));
        let slice_selection = Rc::new(RefCell::new(None));
        let slice_controls = Rc::new(RefCell::new(None));
        let on_slice_change: Rc<RefCell<Option<Box<dyn Fn(SliceSelection)>>>> = Rc::new(RefCell::new(None));
        let active_selection = Rc::new(RefCell::new(None));

        let data_weak = Rc::downgrade(&data);
        let mode_weak = Rc::downgrade(&display_mode);
        drawing_area.set_draw_func(move |_, cr, width, height| {
            if let (Some(data_rc), Some(mode_rc)) = (data_weak.upgrade(), mode_weak.upgrade()) {
                if let Some(heatmap_data) = data_rc.borrow().as_ref() {
                    match *mode_rc.borrow() {
                        DisplayMode::Heatmap => draw_heatmap(cr, width, height, heatmap_data),
                        DisplayMode::Histogram => draw_histogram(cr, width, height, heatmap_data),
                        DisplayMode::LinePlot => draw_line_plot(cr, width, height, heatmap_data),
                    }
                } else {
                    // Draw empty state
                    draw_empty_state(cr, width, height);
                }
            }
        });

        // Create tooltip label
        let tooltip_label = gtk::Label::new(None);
        drawing_area.set_tooltip_text(None);

        // Add motion controller for mouse tracking
        let motion_controller = gtk::EventControllerMotion::new();
        let data_for_motion = Rc::clone(&data);
        let mode_for_motion = Rc::clone(&display_mode);
        let tooltip_label_clone = tooltip_label.clone();
        let drawing_area_weak = drawing_area.downgrade();

        motion_controller.connect_motion(move |_, x, y| {
            if let Some(da) = drawing_area_weak.upgrade() {
                let width = da.width() as f64;
                let height = da.height() as f64;

                if let Some(heatmap_data) = data_for_motion.borrow().as_ref() {
                    let tooltip_text = match *mode_for_motion.borrow() {
                        DisplayMode::Heatmap => {
                            compute_heatmap_tooltip(x, y, width, height, heatmap_data)
                        }
                        DisplayMode::Histogram => {
                            compute_histogram_tooltip(x, y, width, height, heatmap_data)
                        }
                        DisplayMode::LinePlot => {
                            compute_line_plot_tooltip(x, y, width, height, heatmap_data)
                        }
                    };

                    if let Some(text) = tooltip_text {
                        da.set_tooltip_text(Some(&text));
                    } else {
                        da.set_tooltip_text(None);
                    }
                }
            }
        });

        drawing_area.add_controller(motion_controller);

        // Add drawing area to container
        container.append(&drawing_area);

        Self {
            container,
            drawing_area,
            data,
            display_mode,
            tooltip_label,
            slice_selection,
            slice_controls,
            on_slice_change,
            active_selection,
        }
    }

    /// Set a callback to be called when the slice selection changes
    pub fn set_on_slice_change<F>(&self, callback: F)
    where
        F: Fn(SliceSelection) + 'static,
    {
        *self.on_slice_change.borrow_mut() = Some(Box::new(callback));
    }

    pub fn widget(&self) -> &GtkBox {
        &self.container
    }

    /// Set the slice selection and update controls
    pub fn set_slice_selection(&self, selection: Option<SliceSelection>) {
        let needs_controls = selection.as_ref().map(|s| s.needs_slicing()).unwrap_or(false);

        // Remove old controls if they exist
        if let Some(old_controls) = self.slice_controls.borrow_mut().take() {
            self.container.remove(&old_controls);
        }

        // Add new controls if needed
        if needs_controls {
            if let Some(ref sel) = selection {
                let controls = self.create_slice_controls(sel.clone());
                self.container.prepend(&controls);
                *self.slice_controls.borrow_mut() = Some(controls);
            }
        }

        *self.slice_selection.borrow_mut() = selection;
    }

    /// Create the UI controls for slice selection
    fn create_slice_controls(&self, selection: SliceSelection) -> GtkBox {
        let controls_box = GtkBox::new(Orientation::Vertical, 6);
        controls_box.set_margin_start(8);
        controls_box.set_margin_end(8);
        controls_box.set_margin_top(8);
        controls_box.set_margin_bottom(8);

        // Add a label
        let label = gtk::Label::new(Some("Slice Controls"));
        label.set_halign(gtk::Align::Start);
        label.add_css_class("heading");
        controls_box.append(&label);

        // Create a slider for each sliceable dimension
        let sliceable = selection.sliceable_dimensions();

        if sliceable.is_empty() {
            let info = gtk::Label::new(Some("No sliceable dimensions"));
            info.add_css_class("dim-label");
            controls_box.append(&info);
            return controls_box;
        }

        // Store the selection in the widget so it stays alive for slider callbacks
        *self.active_selection.borrow_mut() = Some(selection.clone());

        for (dim_idx, dim_name, dim_size, current_idx) in sliceable {
            let row = GtkBox::new(Orientation::Horizontal, 12);

            // Dimension label
            let dim_label = gtk::Label::new(Some(&format!("{} ({}):", dim_name, dim_size)));
            dim_label.set_width_chars(15);
            dim_label.set_halign(gtk::Align::Start);
            row.append(&dim_label);

            // Slider
            let slider = Scale::with_range(Orientation::Horizontal, 0.0, (dim_size - 1) as f64, 1.0);
            slider.set_hexpand(true);
            slider.set_draw_value(true);
            slider.set_value_pos(gtk::PositionType::Right);
            slider.set_digits(0);

            // Simple approach: set value first, then connect handler
            let active_sel_weak = Rc::downgrade(&self.active_selection);
            let callback_weak = Rc::downgrade(&self.on_slice_change);
            let slice_selection_weak = Rc::downgrade(&self.slice_selection);

            // Set initial value BEFORE connecting handler
            slider.set_value(current_idx as f64);

            // Now connect the handler
            slider.connect_value_changed(move |scale| {
                let new_idx = scale.value() as u64;
                eprintln!("SLIDER CHANGED: dim={}, value={}", dim_idx, new_idx);

                // Add defensive checks for all weak references
                let Some(active_sel_rc) = active_sel_weak.upgrade() else {
                    eprintln!("  -> ERROR: active_sel_weak upgrade failed!");
                    return;
                };

                let mut active_sel_guard = active_sel_rc.borrow_mut();
                let Some(ref mut sel_guard) = active_sel_guard.as_mut() else {
                    eprintln!("  -> ERROR: active_selection is None!");
                    return;
                };

                match sel_guard.set_fixed_index(dim_idx, new_idx) {
                    Ok(()) => {
                        eprintln!("  -> set_fixed_index OK");

                        // Clone the selection before releasing the borrow
                        let sel_clone = sel_guard.clone();

                        // Update the stored slice selection
                        if let Some(sel_storage) = slice_selection_weak.upgrade() {
                            *sel_storage.borrow_mut() = Some(sel_clone.clone());
                        }

                        // Call the callback to reload data
                        if let Some(callback_rc) = callback_weak.upgrade() {
                            if let Some(ref callback) = *callback_rc.borrow() {
                                eprintln!("  -> Calling reload callback");
                                callback(sel_clone);
                            } else {
                                eprintln!("  -> WARNING: Callback is None - this might be normal");
                            }
                        } else {
                            eprintln!("  -> WARNING: callback_weak upgrade failed - popover might be closing");
                        }
                    }
                    Err(e) => {
                        eprintln!("  -> set_fixed_index FAILED: {}", e);
                    }
                }
            });

            eprintln!("Slider initialization completed for dimension {}", dim_idx);

            row.append(&slider);
            controls_box.append(&row);
        }

        // Add keyboard hint
        let hint = gtk::Label::new(Some("Tip: Use arrow keys to navigate slices"));
        hint.add_css_class("dim-label");
        hint.set_halign(gtk::Align::Start);
        hint.set_margin_top(4);
        controls_box.append(&hint);

        controls_box
    }

    /// Set the tensor data to visualize
    /// Automatically downsamples intelligently based on display resolution
    pub fn set_data(&self, values: Vec<f32>, original_shape: &[u64]) {
        // Check if tensor is too large for reasonable visualization
        let total_elements = original_shape.iter().product::<u64>() as usize;
        if total_elements > 50_000_000 { // 50M elements limit
            eprintln!("Warning: Tensor with {} elements is very large, applying aggressive downsampling", total_elements);
        }

        let heatmap_data = prepare_heatmap_data(values, original_shape);

        if let Some(data) = &heatmap_data {
            // Verify the resulting heatmap data isn't too large for the widget
            let heatmap_pixels = data.width * data.height;
            if heatmap_pixels > 2_000_000 { // 2M pixel limit for heatmap
                eprintln!("Warning: Heatmap resolution {}x{} ({} pixels) may cause performance issues",
                    data.width, data.height, heatmap_pixels);
            }

            // Calculate optimal size for the drawing area (add space for legend)
            let aspect_ratio = data.width as f64 / data.height as f64;
            let legend_space = 40.0;

            let (content_width, content_height) = if aspect_ratio > 1.0 {
                // Wider than tall - cap at viewport width
                let w = VIEWPORT_WIDTH.min(data.width as f64 * MIN_CELL_SIZE).min(MAX_DISPLAY_WIDTH);
                let h = (w / aspect_ratio).min(VIEWPORT_HEIGHT - legend_space).min(MAX_DISPLAY_HEIGHT - legend_space);
                (w as i32, h as i32 + legend_space as i32)
            } else {
                // Taller than wide or square - cap at viewport height
                let h = (VIEWPORT_HEIGHT - legend_space).min(data.height as f64 * MIN_CELL_SIZE).min(MAX_DISPLAY_HEIGHT - legend_space);
                let w = (h * aspect_ratio).min(VIEWPORT_WIDTH).min(MAX_DISPLAY_WIDTH);
                (w as i32, h as i32 + legend_space as i32)
            };

            // Apply strict size constraints to prevent layout crashes
            let final_width = content_width.max(150).min((VIEWPORT_WIDTH * 1.2) as i32).min(MAX_DISPLAY_WIDTH as i32);
            let final_height = content_height.max(200).min((VIEWPORT_HEIGHT * 1.2) as i32 + 40).min((MAX_DISPLAY_HEIGHT + 40.0) as i32);

            self.drawing_area.set_content_width(final_width);
            self.drawing_area.set_content_height(final_height);

            eprintln!("Set heatmap size: {}x{} for tensor shape {:?}", final_width, final_height, original_shape);
        } else {
            eprintln!("Error: Failed to prepare heatmap data for tensor with shape {:?}", original_shape);
            // Set minimal size to prevent crashes
            self.drawing_area.set_content_width(200);
            self.drawing_area.set_content_height(150);
        }

        *self.data.borrow_mut() = heatmap_data;
        self.drawing_area.queue_draw();
    }

    /// Get computed statistics from the current data
    pub fn get_statistics(&self) -> Option<TensorStatistics> {
        self.data.borrow().as_ref().map(|data| {
            compute_statistics(&data.values)
        })
    }

    pub fn clear(&self) {
        *self.data.borrow_mut() = None;
        self.drawing_area.queue_draw();
    }

    /// Toggle between heatmap/histogram display modes
    pub fn toggle_display_mode(&self) {
        let mut mode = self.display_mode.borrow_mut();
        *mode = match *mode {
            DisplayMode::Heatmap => DisplayMode::Histogram,
            DisplayMode::Histogram => DisplayMode::Heatmap,
            DisplayMode::LinePlot => DisplayMode::Histogram,
        };
        self.drawing_area.queue_draw();
    }

    /// Set specific display mode
    pub fn set_display_mode(&self, show_histogram: bool) {
        *self.display_mode.borrow_mut() = if show_histogram {
            DisplayMode::Histogram
        } else {
            DisplayMode::Heatmap
        };
        self.drawing_area.queue_draw();
    }

    /// Set display mode to line plot for 1D tensors
    pub fn set_display_mode_line_plot(&self) {
        *self.display_mode.borrow_mut() = DisplayMode::LinePlot;
        self.drawing_area.queue_draw();
    }

    /// Check if currently in line plot mode
    pub fn is_line_plot_mode(&self) -> bool {
        *self.display_mode.borrow() == DisplayMode::LinePlot
    }

    /// Check if currently in heatmap mode
    pub fn is_heatmap_mode(&self) -> bool {
        *self.display_mode.borrow() == DisplayMode::Heatmap
    }

    /// Check if currently in histogram mode
    pub fn is_histogram_mode(&self) -> bool {
        *self.display_mode.borrow() == DisplayMode::Histogram
    }

    /// Get current display mode as a string (for UI labels)
    pub fn get_display_mode_label(&self) -> String {
        match *self.display_mode.borrow() {
            DisplayMode::Heatmap => "Heatmap".to_string(),
            DisplayMode::Histogram => "Histogram".to_string(),
            DisplayMode::LinePlot => "Line Plot".to_string(),
        }
    }
}

/// Prepare and downsample tensor data for heatmap visualization
fn prepare_heatmap_data(values: Vec<f32>, shape: &[u64]) -> Option<HeatmapData> {
    if values.is_empty() || shape.is_empty() {
        return None;
    }

    // Determine 2D layout from shape and apply appropriate downsampling
    let (width, height, downsampled_values) = match shape.len() {
        1 => {
            // 1D tensor - apply peak-preserving downsampling for large tensors
            let total = shape[0] as usize;
            let downsampled = if total > 10_000 {
                downsample_1d_peak_preserving(values, 10_000.min(total))
            } else {
                values
            };
            (downsampled.len(), 1, downsampled)
        },
        2 => {
            let width = shape[1] as usize;
            let height = shape[0] as usize;
            // Downsample if necessary
            if width > MAX_HEATMAP_DIMENSION || height > MAX_HEATMAP_DIMENSION {
                downsample_data(&values, width, height)
            } else {
                (width, height, values)
            }
        },
        3 => {
            // For 3D tensors (e.g., [batch, height, width]), flatten batch dimension
            let width = shape[2] as usize;
            let height = (shape[0] * shape[1]) as usize;
            if width > MAX_HEATMAP_DIMENSION || height > MAX_HEATMAP_DIMENSION {
                downsample_data(&values, width, height)
            } else {
                (width, height, values)
            }
        },
        _ => {
            // For higher dimensional tensors, flatten all but last two dimensions
            let width = shape[shape.len() - 1] as usize;
            let height: usize = shape[..shape.len() - 1].iter().map(|&d| d as usize).product();
            if width > MAX_HEATMAP_DIMENSION || height > MAX_HEATMAP_DIMENSION {
                downsample_data(&values, width, height)
            } else {
                (width, height, values)
            }
        }
    };

    // Calculate min and max for color mapping
    let min_value = downsampled_values.iter()
        .copied()
        .filter(|v| v.is_finite())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let max_value = downsampled_values.iter()
        .copied()
        .filter(|v| v.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(1.0);

    Some(HeatmapData {
        values: downsampled_values,
        width,
        height,
        min_value,
        max_value,
    })
}

/// Intelligent downsampling that preserves tensor structure and important features
fn downsample_data(values: &[f32], width: usize, height: usize) -> (usize, usize, Vec<f32>) {
    let total_pixels = width * height;

    // Calculate target resolution based on content and display constraints
    let (max_width, max_height) = calculate_target_resolution(width, height, total_pixels);

    // If no downsampling needed, return as-is
    if width <= max_width && height <= max_height {
        return (width, height, values.to_vec());
    }

    // Calculate adaptive scaling factors
    let scale_x = (width as f64 / max_width as f64).ceil() as usize;
    let scale_y = (height as f64 / max_height as f64).ceil() as usize;

    let new_width = (width + scale_x - 1) / scale_x;
    let new_height = (height + scale_y - 1) / scale_y;

    let mut downsampled = vec![0.0f32; new_width * new_height];

    // Use intelligent downsampling strategy based on scale factor
    if scale_x <= 2 && scale_y <= 2 {
        // Light downsampling: use averaging for smooth results
        average_downsample(values, width, height, scale_x, scale_y, &mut downsampled);
    } else if scale_x <= 4 && scale_y <= 4 {
        // Moderate downsampling: use importance-weighted sampling
        importance_weighted_downsample(values, width, height, scale_x, scale_y, &mut downsampled);
    } else {
        // Heavy downsampling: use adaptive stratified sampling
        adaptive_stratified_downsample(values, width, height, scale_x, scale_y, &mut downsampled);
    }

    (new_width, new_height, downsampled)
}

/// Calculate target resolution based on tensor characteristics and display constraints
fn calculate_target_resolution(width: usize, height: usize, total_pixels: usize) -> (usize, usize) {
    match total_pixels {
        // Small tensors: keep original resolution but respect widget limits
        0..=100_000 => {
            (width.min(MAX_HEATMAP_DIMENSION), height.min(MAX_HEATMAP_DIMENSION))
        },

        // Medium tensors: moderate resolution but preserve aspect ratio
        100_001..=1_000_000 => {
            let max_dim = 1000.min(MAX_HEATMAP_DIMENSION);
            let aspect_ratio = width as f64 / height as f64;

            if aspect_ratio > 1.5 {
                // Wide tensor
                let new_width = max_dim;
                let new_height = (max_dim as f64 / aspect_ratio) as usize;
                (new_width, new_height.max(50)) // Minimum height to avoid tiny strips
            } else if aspect_ratio < 0.67 {
                // Tall tensor
                let new_height = max_dim;
                let new_width = (max_dim as f64 * aspect_ratio) as usize;
                (new_width.max(50), new_height) // Minimum width to avoid tiny strips
            } else {
                // Balanced: square-ish
                (max_dim, max_dim)
            }
        },

        // Large tensors: capped at reasonable widget size
        1_000_001..=10_000_000 => {
            let max_width = (VIEWPORT_WIDTH / MIN_CELL_SIZE) as usize;
            let max_height = (VIEWPORT_HEIGHT / MIN_CELL_SIZE) as usize;
            let aspect_ratio = width as f64 / height as f64;

            if aspect_ratio > 2.0 {
                // Very wide: use viewport width
                (max_width, ((max_width as f64 / aspect_ratio) as usize).max(30))
            } else if aspect_ratio < 0.5 {
                // Very tall: use viewport height
                (((max_height as f64 * aspect_ratio) as usize).max(30), max_height)
            } else {
                // Balanced: fit within viewport
                (max_width, max_height)
            }
        },

        // Very large tensors: strict viewport limits
        _ => {
            let viewport_width = (VIEWPORT_WIDTH / MIN_CELL_SIZE) as usize;
            let viewport_height = (VIEWPORT_HEIGHT / MIN_CELL_SIZE) as usize;

            // Always cap to viewport dimensions
            (viewport_width, viewport_height)
        }
    }
}

/// Peak-preserving downsampling for 1D tensors (line plots)
/// Preserves local min/max values to maintain visual features
/// Size-based strategy:
/// - ≤10K elements: No downsampling
/// - 10K-100K: Uniform sampling (simple stride)
/// - 100K-1M: Peak-preserving (keep min + max from each bucket)
/// - >1M: Aggressive peak-preserving with larger buckets
pub fn downsample_1d_peak_preserving(values: Vec<f32>, target_size: usize) -> Vec<f32> {
    let num_elements = values.len();

    // Small tensors: no downsampling
    if num_elements <= target_size {
        return values;
    }

    // Very large tensors (>1M elements): aggressive peak-preserving
    if num_elements > 1_000_000 {
        return downsample_1d_aggressive(values, target_size);
    }

    // Large tensors (100K-1M): peak-preserving downsampling
    if num_elements > 100_000 {
        return downsample_1d_peak_bucket(values, target_size);
    }

    // Medium tensors (10K-100K): simple uniform sampling
    downsample_1d_uniform(values, target_size)
}

/// Uniform sampling for medium-sized 1D tensors
fn downsample_1d_uniform(values: Vec<f32>, target_size: usize) -> Vec<f32> {
    let stride = (values.len() + target_size - 1) / target_size;
    values.into_iter()
        .enumerate()
        .filter(|(i, _)| i % stride == 0)
        .map(|(_, v)| v)
        .take(target_size)
        .collect()
}

/// Peak-preserving bucket downsampling for large 1D tensors
/// Keeps both min and max from each bucket to preserve peaks and valleys
fn downsample_1d_peak_bucket(values: Vec<f32>, target_size: usize) -> Vec<f32> {
    let num_elements = values.len();
    let bucket_size = (num_elements + target_size - 1) / target_size;
    let num_buckets = (num_elements + bucket_size - 1) / bucket_size;

    let mut result = Vec::with_capacity(num_buckets * 2);

    for bucket_idx in 0..num_buckets {
        let start = bucket_idx * bucket_size;
        let end = (start + bucket_size).min(num_elements);

        if start >= num_elements {
            break;
        }

        let bucket = &values[start..end];

        // Find min and max in this bucket
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &val in bucket {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Add min and max (preserves peaks and valleys)
        if min_val.is_finite() {
            result.push(min_val);
        }
        if max_val.is_finite() && max_val != min_val {
            result.push(max_val);
        }
    }

    // If we still have too many points, do a second pass of uniform sampling
    if result.len() > target_size {
        let stride = (result.len() + target_size - 1) / target_size;
        result = result.into_iter()
            .enumerate()
            .filter(|(i, _)| i % stride == 0)
            .map(|(_, v)| v)
            .collect();
    }

    result
}

/// Aggressive peak-preserving for very large 1D tensors (>1M elements)
/// Uses larger buckets with adaptive sampling within each bucket
fn downsample_1d_aggressive(values: Vec<f32>, target_size: usize) -> Vec<f32> {
    let num_elements = values.len();
    let bucket_size = ((num_elements / target_size) * 4).max(100); // Larger buckets
    let num_buckets = (num_elements + bucket_size - 1) / bucket_size;

    let mut result = Vec::with_capacity(target_size);

    for bucket_idx in 0..num_buckets {
        let start = bucket_idx * bucket_size;
        let end = (start + bucket_size).min(num_elements);

        if start >= num_elements {
            break;
        }

        let bucket = &values[start..end];

        // Sample 5 points from each bucket: min, max, and 3 intermediate
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        let mut count = 0usize;

        for &val in bucket {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
                sum += val;
                count += 1;
            }
        }

        // Add min, max, and average
        if min_val.is_finite() {
            result.push(min_val);
        }
        if count > 0 {
            result.push(sum / count as f32);
        }
        if max_val.is_finite() && max_val != min_val {
            result.push(max_val);
        }
    }

    // Final uniform sampling if still too large
    if result.len() > target_size {
        let stride = (result.len() + target_size - 1) / target_size;
        result = result.into_iter()
            .enumerate()
            .filter(|(i, _)| i % stride == 0)
            .map(|(_, v)| v)
            .collect();
    }

    result
}

/// Simple averaging downsampling for light scaling (best for small scale factors)
fn average_downsample(
    values: &[f32],
    width: usize,
    height: usize,
    scale_x: usize,
    scale_y: usize,
    output: &mut [f32]
) {
    let new_width = output.len() as usize / ((height + scale_y - 1) / scale_y);

    for out_y in 0..output.len() / new_width {
        for out_x in 0..new_width {
            let in_x_start = out_x * scale_x;
            let in_y_start = out_y * scale_y;
            let in_x_end = (in_x_start + scale_x).min(width);
            let in_y_end = (in_y_start + scale_y).min(height);

            let mut sum = 0.0f32;
            let mut count = 0;

            for in_y in in_y_start..in_y_end {
                for in_x in in_x_start..in_x_end {
                    let idx = in_y * width + in_x;
                    if idx < values.len() && values[idx].is_finite() {
                        sum += values[idx];
                        count += 1;
                    }
                }
            }

            let out_idx = out_y * new_width + out_x;
            output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }
}

/// Importance-weighted downsampling that preserves significant features
fn importance_weighted_downsample(
    values: &[f32],
    width: usize,
    height: usize,
    scale_x: usize,
    scale_y: usize,
    output: &mut [f32]
) {
    let new_width = output.len() as usize / ((height + scale_y - 1) / scale_y);

    for out_y in 0..output.len() / new_width {
        for out_x in 0..new_width {
            let in_x_start = out_x * scale_x;
            let in_y_start = out_y * scale_y;
            let in_x_end = (in_x_start + scale_x).min(width);
            let in_y_end = (in_y_start + scale_y).min(height);

            let mut weighted_sum = 0.0f32;
            let mut weight_total = 0.0f32;

            for in_y in in_y_start..in_y_end {
                for in_x in in_x_start..in_x_end {
                    let idx = in_y * width + in_x;
                    if idx < values.len() && values[idx].is_finite() {
                        let value = values[idx];
                        // Use magnitude-based weighting to preserve significant values
                        let weight = 1.0 + value.abs().ln_1p(); // ln(1 + |value|)
                        weighted_sum += value * weight;
                        weight_total += weight;
                    }
                }
            }

            let out_idx = out_y * new_width + out_x;
            output[out_idx] = if weight_total > 0.0 { weighted_sum / weight_total } else { 0.0 };
        }
    }
}

/// Adaptive stratified downsampling for very large tensors
fn adaptive_stratified_downsample(
    values: &[f32],
    width: usize,
    height: usize,
    scale_x: usize,
    scale_y: usize,
    output: &mut [f32]
) {
    let new_width = output.len() as usize / ((height + scale_y - 1) / scale_y);

    // For heavy downsampling, use a combination of sampling strategies
    for out_y in 0..output.len() / new_width {
        for out_x in 0..new_width {
            let in_x_start = out_x * scale_x;
            let in_y_start = out_y * scale_y;
            let in_x_end = (in_x_start + scale_x).min(width);
            let in_y_end = (in_y_start + scale_y).min(height);

            // Collect samples from the region
            let mut samples = Vec::new();

            // Always include corners (structural importance)
            for &(y, x) in &[(in_y_start, in_x_start), (in_y_start, in_x_end-1),
                             (in_y_end-1, in_x_start), (in_y_end-1, in_x_end-1)] {
                if y < height && x < width {
                    let idx = y * width + x;
                    if idx < values.len() && values[idx].is_finite() {
                        samples.push(values[idx]);
                    }
                }
            }

            // Include center (representative)
            let center_y = (in_y_start + in_y_end) / 2;
            let center_x = (in_x_start + in_x_end) / 2;
            if center_y < height && center_x < width {
                let idx = center_y * width + center_x;
                if idx < values.len() && values[idx].is_finite() {
                    samples.push(values[idx]);
                }
            }

            // Add deterministic samples within the region for statistical representation
            let random_samples = 3.min((in_x_end - in_x_start) * (in_y_end - in_y_start) / 4);
            for i in 0..random_samples {
                // Use simple hash-based deterministic sampling
                let seed = (out_y * new_width + out_x) * 7 + i * 13;
                let rand_y = in_y_start + (seed % (in_y_end - in_y_start).max(1));
                let rand_x = in_x_start + ((seed * 17) % (in_x_end - in_x_start).max(1));
                if rand_y < height && rand_x < width {
                    let idx = rand_y * width + rand_x;
                    if idx < values.len() && values[idx].is_finite() {
                        samples.push(values[idx]);
                    }
                }
            }

            // Use weighted median of samples for robustness
            let out_idx = out_y * new_width + out_x;
            if samples.is_empty() {
                output[out_idx] = 0.0;
            } else {
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                output[out_idx] = samples[samples.len() / 2]; // Median
            }
        }
    }
}

fn draw_heatmap(cr: &gtk::cairo::Context, width: i32, height: i32, data: &HeatmapData) {
    let width = width as f64;
    let height = height as f64;

    // Reserve space for legend at bottom
    let legend_height = 40.0;
    let heatmap_height = height - legend_height;

    // Calculate cell dimensions
    let cell_width = width / data.width as f64;
    let cell_height = heatmap_height / data.height as f64;

    let value_range = (data.max_value - data.min_value).max(0.0001);

    // Draw each cell
    for y in 0..data.height {
        for x in 0..data.width {
            let idx = y * data.width + x;
            if idx >= data.values.len() {
                continue;
            }

            let value = data.values[idx];
            if !value.is_finite() {
                continue;
            }

            // Normalize value to 0-1 range
            let normalized = ((value - data.min_value) / value_range).clamp(0.0, 1.0);

            // Apply color mapping using memory visualizer gradient (green -> blue)
            let (r, g, b) = memory_viz_color_gradient(normalized as f64);

            cr.set_source_rgb(r, g, b);
            cr.rectangle(
                x as f64 * cell_width,
                y as f64 * cell_height,
                cell_width + 0.5,  // Add small overlap to prevent gaps
                cell_height + 0.5, // Add small overlap to prevent gaps
            );
            cr.fill().unwrap();
        }
    }

    // Grid lines removed for cleaner visualization

    // Draw colormap legend at bottom
    draw_colormap_legend(cr, width, heatmap_height, legend_height, data.min_value, data.max_value);
}

fn draw_empty_state(cr: &gtk::cairo::Context, width: i32, height: i32) {
    let width = width as f64;
    let height = height as f64;

    // Draw a light gray background
    cr.set_source_rgb(0.95, 0.95, 0.95);
    cr.rectangle(0.0, 0.0, width, height);
    cr.fill().unwrap();

    // Draw text
    cr.set_source_rgb(0.5, 0.5, 0.5);
    cr.select_font_face("Sans", gtk::cairo::FontSlant::Normal, gtk::cairo::FontWeight::Normal);
    cr.set_font_size(14.0);

    let text = "No data to display";
    let extents = cr.text_extents(text).unwrap();
    cr.move_to(
        (width - extents.width()) / 2.0,
        (height + extents.height()) / 2.0,
    );
    cr.show_text(text).unwrap();
}

/// Map a normalized value (0-1) to an RGB color using a perceptually uniform colormap
/// Similar to matplotlib's "turbo" or "viridis" colormap
fn value_to_color(value: f64) -> (f64, f64, f64) {
    // Five-color gradient: blue -> cyan -> green -> yellow -> red
    let value = value.clamp(0.0, 1.0);

    if value < 0.25 {
        // Blue to cyan
        let t = value / 0.25;
        let r = 0.0;
        let g = t * 0.5;
        let b = 1.0;
        (r, g, b)
    } else if value < 0.5 {
        // Cyan to green
        let t = (value - 0.25) / 0.25;
        let r = 0.0;
        let g = 0.5 + t * 0.5;
        let b = 1.0 - t;
        (r, g, b)
    } else if value < 0.75 {
        // Green to yellow
        let t = (value - 0.5) / 0.25;
        let r = t;
        let g = 1.0;
        let b = 0.0;
        (r, g, b)
    } else {
        // Yellow to red
        let t = (value - 0.75) / 0.25;
        let r = 1.0;
        let g = 1.0 - t;
        let b = 0.0;
        (r, g, b)
    }
}

/// Draw a colormap legend showing the value range
fn draw_colormap_legend(cr: &gtk::cairo::Context, width: f64, y_offset: f64, legend_height: f64, min_val: f32, max_val: f32) {
    let margin = 12.0;
    let bar_height = 12.0;
    let bar_y = y_offset + 8.0;
    let bar_width = width - 2.0 * margin;
    let num_segments = 100;
    let segment_width = bar_width / num_segments as f64;

    // Draw gradient bar
    for i in 0..num_segments {
        let t = i as f64 / (num_segments - 1) as f64;
        let (r, g, b) = memory_viz_color_gradient(t);
        cr.set_source_rgb(r, g, b);
        cr.rectangle(
            margin + i as f64 * segment_width,
            bar_y,
            segment_width + 0.5,  // Add small overlap to prevent gaps
            bar_height,
        );
        cr.fill().unwrap();
    }

    // Border removed for cleaner visualization

    // Draw min/max labels
    cr.set_source_rgba(0.8, 0.8, 0.8, 0.9);
    cr.select_font_face("Sans", gtk::cairo::FontSlant::Normal, gtk::cairo::FontWeight::Normal);
    cr.set_font_size(9.0);

    let min_text = format!("{:.4}", min_val);
    cr.move_to(margin, bar_y + bar_height + 12.0);
    cr.show_text(&min_text).unwrap();

    let max_text = format!("{:.4}", max_val);
    let extents = cr.text_extents(&max_text).unwrap();
    cr.move_to(width - margin - extents.width(), bar_y + bar_height + 12.0);
    cr.show_text(&max_text).unwrap();
}

/// Compute comprehensive statistics for tensor values (optimized)
fn compute_statistics(values: &[f32]) -> TensorStatistics {
    if values.is_empty() {
        return TensorStatistics {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            sparsity: 0.0,
            median: 0.0,
        };
    }

    // Single-pass statistics computation (no intermediate Vec allocation)
    let mut sum = 0.0f32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut count = 0usize;

    for &value in values {
        if value.is_finite() {
            sum += value;
            min = min.min(value);
            max = max.max(value);
            count += 1;
        }
    }

    if count == 0 {
        return TensorStatistics {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            sparsity: 100.0,
            median: 0.0,
        };
    }

    let mean = sum / count as f32;

    // Second pass for variance and sparsity
    let mut variance = 0.0f32;
    let mut near_zero_count = 0usize;
    let threshold = max.abs().max(min.abs()) * 0.001;

    for &value in values {
        if value.is_finite() {
            let diff = value - mean;
            variance += diff * diff;

            if value.abs() < threshold {
                near_zero_count += 1;
            }
        }
    }

    let std_dev = (variance / count as f32).sqrt();
    let sparsity = (near_zero_count as f32 / count as f32) * 100.0;

    // Optimized median computation - use sampling for large arrays
    let median = if count > 10000 {
        // Sample-based approximation for large tensors (avoids sorting overhead)
        let sample_size = 1000.min(count);
        let step = count / sample_size;
        let mut sample: Vec<f32> = values.iter()
            .copied()
            .enumerate()
            .filter(|(i, v)| i % step == 0 && v.is_finite())
            .map(|(_, v)| v)
            .take(sample_size)
            .collect();

        if !sample.is_empty() {
            let mid = sample.len() / 2;
            *sample.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1
        } else {
            mean
        }
    } else {
        // Exact median for smaller datasets (using select_nth_unstable for O(n) time)
        let mut sorted: Vec<f32> = values.iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();

        if !sorted.is_empty() {
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                sorted.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                *sorted.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1
            }
        } else {
            mean
        }
    };

    TensorStatistics {
        mean,
        std_dev,
        min,
        max,
        sparsity,
        median,
    }
}

/// Draw a histogram of tensor values
fn draw_histogram(cr: &gtk::cairo::Context, width: i32, height: i32, data: &HeatmapData) {
    let width = width as f64;
    let height = height as f64;
    let margin = 20.0;
    let bottom_margin = 50.0; // Extra space for labels at bottom
    let plot_width = width - 2.0 * margin;
    let plot_height = height - margin - bottom_margin;

    if plot_width <= 0.0 || plot_height <= 0.0 {
        // Not enough space to draw
        draw_empty_state(cr, width as i32, height as i32);
        return;
    }

    // Compute histogram bins
    let num_bins = 50;
    let mut bins = vec![0usize; num_bins];

    let value_range = (data.max_value - data.min_value).max(0.0001);

    for &value in &data.values {
        if !value.is_finite() {
            continue;
        }

        let normalized = ((value - data.min_value) / value_range).clamp(0.0, 1.0);
        let bin_idx = ((normalized * (num_bins - 1) as f32) as usize).min(num_bins - 1);
        bins[bin_idx] += 1;
    }

    // Find max bin count for scaling
    let max_count = *bins.iter().max().unwrap_or(&1);

    if max_count == 0 {
        draw_empty_state(cr, width as i32, height as i32);
        return;
    }

    // Draw background
    cr.set_source_rgba(0.0, 0.0, 0.0, 0.05);
    cr.rectangle(margin, margin, plot_width, plot_height);
    cr.fill().unwrap();

    // Draw histogram bars
    let bar_width = plot_width / num_bins as f64;

    for (i, &count) in bins.iter().enumerate() {
        if count == 0 {
            continue;
        }

        let bar_height = (count as f64 / max_count as f64) * plot_height;
        let x = margin + i as f64 * bar_width;
        let y = margin + plot_height - bar_height;

        // Use a single neutral color for histogram bars
        // The histogram shows frequency distribution, not value magnitude
        // Use a light blue-gray color that matches the dark theme
        cr.set_source_rgba(0.45, 0.55, 0.65, 0.85);
        cr.rectangle(x, y, bar_width * 0.9, bar_height);
        cr.fill().unwrap();
    }

    // Draw axes
    cr.set_source_rgba(0.5, 0.5, 0.5, 0.8);
    cr.set_line_width(1.5);

    // Y-axis
    cr.move_to(margin, margin);
    cr.line_to(margin, margin + plot_height);
    cr.stroke().unwrap();

    // X-axis
    cr.move_to(margin, margin + plot_height);
    cr.line_to(margin + plot_width, margin + plot_height);
    cr.stroke().unwrap();

    // Draw labels
    cr.set_source_rgba(0.8, 0.8, 0.8, 0.9);
    cr.select_font_face("Sans", gtk::cairo::FontSlant::Normal, gtk::cairo::FontWeight::Normal);
    cr.set_font_size(9.0);

    // Min value label
    let min_text = format!("{:.3}", data.min_value);
    cr.move_to(margin, height - 8.0);
    cr.show_text(&min_text).unwrap();

    // Max value label
    let max_text = format!("{:.3}", data.max_value);
    let extents = cr.text_extents(&max_text).unwrap();
    cr.move_to(width - margin - extents.width(), height - 8.0);
    cr.show_text(&max_text).unwrap();

    // Y-axis label (count)
    cr.save().unwrap();
    cr.move_to(8.0, margin + plot_height / 2.0);
    cr.rotate(-std::f64::consts::PI / 2.0);
    cr.show_text("Frequency").unwrap();
    cr.restore().unwrap();

    // Title
    cr.set_font_size(11.0);
    cr.set_source_rgba(0.9, 0.9, 0.9, 1.0);
    let title = "Value Distribution";
    let title_extents = cr.text_extents(title).unwrap();
    cr.move_to((width - title_extents.width()) / 2.0, margin - 5.0);
    cr.show_text(title).unwrap();
}

/// Draw a line plot for 1D tensor values
fn draw_line_plot(cr: &gtk::cairo::Context, width: i32, height: i32, data: &HeatmapData) {
    let width = width as f64;
    let height = height as f64;
    let margin = 40.0;
    let bottom_margin = 55.0;
    let left_margin = 50.0;
    let right_margin = 20.0;
    let top_margin = 30.0;
    let plot_width = width - left_margin - right_margin;
    let plot_height = height - top_margin - bottom_margin;

    if plot_width <= 0.0 || plot_height <= 0.0 {
        draw_empty_state(cr, width as i32, height as i32);
        return;
    }

    let num_points = data.values.len();
    if num_points == 0 {
        draw_empty_state(cr, width as i32, height as i32);
        return;
    }

    let value_range = (data.max_value - data.min_value).max(0.0001);

    // Draw background
    cr.set_source_rgba(0.0, 0.0, 0.0, 0.05);
    cr.rectangle(left_margin, top_margin, plot_width, plot_height);
    cr.fill().unwrap();

    // Draw grid lines (horizontal)
    cr.set_source_rgba(0.3, 0.3, 0.3, 0.3);
    cr.set_line_width(0.5);
    for i in 0..=4 {
        let y = top_margin + (plot_height * i as f64 / 4.0);
        cr.move_to(left_margin, y);
        cr.line_to(left_margin + plot_width, y);
        cr.stroke().unwrap();
    }

    // Draw grid lines (vertical)
    for i in 0..=10 {
        let x = left_margin + (plot_width * i as f64 / 10.0);
        cr.move_to(x, top_margin);
        cr.line_to(x, top_margin + plot_height);
        cr.stroke().unwrap();
    }

    // Draw the line plot using the memory visualizer color gradient
    cr.set_line_width(1.5);
    cr.set_source_rgba(0.46, 0.78, 0.39, 1.0); // Green from memory_viz_color_gradient

    // Build the path
    cr.move_to(
        left_margin,
        top_margin + plot_height - ((data.values[0] - data.min_value) / value_range as f32) as f64 * plot_height,
    );

    for (i, &value) in data.values.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }

        let x = left_margin + (i as f64 / (num_points - 1) as f64) * plot_width;
        let normalized = ((value - data.min_value) / value_range as f32).clamp(0.0, 1.0);
        let y = top_margin + plot_height - (normalized as f64 * plot_height);

        cr.line_to(x, y);
    }

    cr.stroke().unwrap();

    // Draw axes
    cr.set_source_rgba(0.5, 0.5, 0.5, 0.8);
    cr.set_line_width(1.5);

    // Y-axis
    cr.move_to(left_margin, top_margin);
    cr.line_to(left_margin, top_margin + plot_height);
    cr.stroke().unwrap();

    // X-axis
    cr.move_to(left_margin, top_margin + plot_height);
    cr.line_to(left_margin + plot_width, top_margin + plot_height);
    cr.stroke().unwrap();

    // Draw labels
    cr.set_source_rgba(0.8, 0.8, 0.8, 0.9);
    cr.select_font_face("Sans", gtk::cairo::FontSlant::Normal, gtk::cairo::FontWeight::Normal);
    cr.set_font_size(9.0);

    // Y-axis labels (value range)
    for i in 0..=4 {
        let t = i as f64 / 4.0;
        let value = data.min_value + (value_range as f32 * t as f32);
        let text = format!("{:.3}", value);
        let extents = cr.text_extents(&text).unwrap();
        let y = top_margin + plot_height - (t * plot_height);
        cr.move_to(left_margin - extents.width() - 8.0, y + extents.height() / 2.0);
        cr.show_text(&text).unwrap();
    }

    // X-axis labels (indices)
    for i in 0..=5 {
        let idx = (num_points as f64 * i as f64 / 5.0).round() as usize;
        let idx = idx.min(num_points - 1);
        let text = format!("{}", idx);
        let extents = cr.text_extents(&text).unwrap();
        let x = left_margin + (i as f64 / 5.0) * plot_width;
        cr.move_to(x - extents.width() / 2.0, height - bottom_margin + 20.0);
        cr.show_text(&text).unwrap();
    }

    // Axis titles
    cr.set_font_size(10.0);
    cr.set_source_rgba(0.9, 0.9, 0.9, 1.0);

    // Y-axis title
    cr.save().unwrap();
    cr.move_to(12.0, top_margin + plot_height / 2.0);
    cr.rotate(-std::f64::consts::PI / 2.0);
    cr.show_text("Value").unwrap();
    cr.restore().unwrap();

    // X-axis title
    let x_title = "Index";
    let x_title_extents = cr.text_extents(x_title).unwrap();
    cr.move_to(left_margin + plot_width / 2.0 - x_title_extents.width() / 2.0, height - 8.0);
    cr.show_text(x_title).unwrap();

    // Title
    cr.set_font_size(11.0);
    let title = format!("1D Tensor Values ({} points)", num_points);
    let title_extents = cr.text_extents(&title).unwrap();
    cr.move_to((width - title_extents.width()) / 2.0, top_margin - 8.0);
    cr.show_text(&title).unwrap();
}

/// Compute tooltip text for line plot mode
/// Returns index and value at mouse position
fn compute_line_plot_tooltip(x: f64, y: f64, width: f64, height: f64, data: &HeatmapData) -> Option<String> {
    let left_margin = 50.0;
    let right_margin = 20.0;
    let top_margin = 30.0;
    let bottom_margin = 55.0;
    let plot_width = width - left_margin - right_margin;
    let plot_height = height - top_margin - bottom_margin;

    // Check if mouse is in the plot area
    if x < left_margin || x > left_margin + plot_width || y < top_margin || y > top_margin + plot_height {
        return None;
    }

    let num_points = data.values.len();
    if num_points == 0 {
        return None;
    }

    // Find the nearest data point based on x position
    let relative_x = x - left_margin;
    let index_ratio = relative_x / plot_width;
    let index = (index_ratio * (num_points - 1) as f64).round() as usize;
    let index = index.min(num_points - 1);

    // Get the value at this index
    let value = data.values.get(index)?;
    if !value.is_finite() {
        return Some(format!("Index: {}\nValue: N/A", index));
    }

    Some(format!("Index: {}\nValue: {:.6}", index, value))
}

/// Compute tooltip text for heatmap mode
/// Returns cell coordinates and value at mouse position
fn compute_heatmap_tooltip(x: f64, y: f64, width: f64, height: f64, data: &HeatmapData) -> Option<String> {
    let legend_height = 40.0;
    let heatmap_height = height - legend_height;

    // Check if mouse is in the heatmap area (not in legend)
    if y >= heatmap_height {
        return None;
    }

    // Calculate cell dimensions
    let cell_width = width / data.width as f64;
    let cell_height = heatmap_height / data.height as f64;

    // Calculate cell coordinates
    let cell_x = (x / cell_width).floor() as usize;
    let cell_y = (y / cell_height).floor() as usize;

    // Check bounds
    if cell_x >= data.width || cell_y >= data.height {
        return None;
    }

    // Get value
    let idx = cell_y * data.width + cell_x;
    if idx >= data.values.len() {
        return None;
    }

    let value = data.values[idx];
    if !value.is_finite() {
        return Some(format!("Position: ({}, {})\nValue: N/A", cell_x, cell_y));
    }

    Some(format!("Position: ({}, {})\nValue: {:.6}", cell_x, cell_y, value))
}

/// Compute tooltip text for histogram mode
/// Returns bin range and frequency at mouse position
fn compute_histogram_tooltip(x: f64, y: f64, width: f64, height: f64, data: &HeatmapData) -> Option<String> {
    let margin = 20.0;
    let bottom_margin = 50.0;
    let plot_width = width - 2.0 * margin;
    let plot_height = height - margin - bottom_margin;

    // Check if mouse is in the plot area
    if x < margin || x > margin + plot_width || y < margin || y > margin + plot_height {
        return None;
    }

    // Calculate which bin we're hovering over
    let num_bins = 50;
    let bar_width = plot_width / num_bins as f64;
    let bin_idx = ((x - margin) / bar_width).floor() as usize;

    if bin_idx >= num_bins {
        return None;
    }

    // Recompute histogram bins to get frequency (same logic as draw_histogram)
    let mut bins = vec![0usize; num_bins];
    let value_range = (data.max_value - data.min_value).max(0.0001);

    for &value in &data.values {
        if !value.is_finite() {
            continue;
        }

        let normalized = ((value - data.min_value) / value_range).clamp(0.0, 1.0);
        let idx = ((normalized * (num_bins - 1) as f32) as usize).min(num_bins - 1);
        bins[idx] += 1;
    }

    let count = bins[bin_idx];

    // Calculate value range for this bin
    let bin_start = data.min_value + (bin_idx as f32 * value_range / num_bins as f32);
    let bin_end = data.min_value + ((bin_idx + 1) as f32 * value_range / num_bins as f32);

    Some(format!(
        "Range: [{:.4}, {:.4})\nFrequency: {} values",
        bin_start, bin_end, count
    ))
}

/// Color gradient matching the memory visualizer (green to blue)
/// Used to maintain visual consistency across the application
/// Uses a smooth multi-stop gradient to avoid color banding
fn memory_viz_color_gradient(t: f64) -> (f64, f64, f64) {
    let t = t.clamp(0.0, 1.0);

    // Multi-stop gradient for smoother transitions:
    // 0.0: Dark green (low values)
    // 0.33: Bright green
    // 0.66: Cyan (transition)
    // 1.0: Bright blue (high values)

    if t < 0.33 {
        // Dark green to bright green
        let local_t = t / 0.33;
        let r = 0.37 + (0.46 - 0.37) * local_t;
        let g = 0.62 + (0.78 - 0.62) * local_t;
        let b = 0.31 + (0.39 - 0.31) * local_t;
        (r, g, b)
    } else if t < 0.66 {
        // Bright green to cyan
        let local_t = (t - 0.33) / 0.33;
        let r = 0.46 + (0.35 - 0.46) * local_t;
        let g = 0.78 + (0.70 - 0.78) * local_t;
        let b = 0.39 + (0.70 - 0.39) * local_t;
        (r, g, b)
    } else {
        // Cyan to bright blue
        let local_t = (t - 0.66) / 0.34;
        let r = 0.35 + (0.25 - 0.35) * local_t;
        let g = 0.70 + (0.59 - 0.70) * local_t;
        let b = 0.70 + (0.95 - 0.70) * local_t;
        (r, g, b)
    }
}
