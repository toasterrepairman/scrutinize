use gtk::prelude::*;
use gtk::DrawingArea;
use std::cell::RefCell;
use std::rc::Rc;

/// Configuration for heatmap rendering
const MAX_HEATMAP_DIMENSION: usize = 200; // Reduced for better performance
const MIN_CELL_SIZE: f64 = 2.0; // Minimum pixel size for each cell
const MAX_DISPLAY_WIDTH: f64 = 400.0; // Maximum widget width
const MAX_DISPLAY_HEIGHT: f64 = 400.0; // Maximum widget height

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
    drawing_area: DrawingArea,
    data: Rc<RefCell<Option<HeatmapData>>>,
    display_mode: Rc<RefCell<DisplayMode>>,
    tooltip_label: gtk::Label,
}

#[derive(Clone, Copy, PartialEq)]
enum DisplayMode {
    Heatmap,
    Histogram,
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
        let drawing_area = DrawingArea::new();
        drawing_area.set_content_width(400);
        drawing_area.set_content_height(400);

        let data = Rc::new(RefCell::new(None));
        let display_mode = Rc::new(RefCell::new(DisplayMode::Heatmap));

        let data_weak = Rc::downgrade(&data);
        let mode_weak = Rc::downgrade(&display_mode);
        drawing_area.set_draw_func(move |_, cr, width, height| {
            if let (Some(data_rc), Some(mode_rc)) = (data_weak.upgrade(), mode_weak.upgrade()) {
                if let Some(heatmap_data) = data_rc.borrow().as_ref() {
                    match *mode_rc.borrow() {
                        DisplayMode::Heatmap => draw_heatmap(cr, width, height, heatmap_data),
                        DisplayMode::Histogram => draw_histogram(cr, width, height, heatmap_data),
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

        Self {
            drawing_area,
            data,
            display_mode,
            tooltip_label,
        }
    }

    pub fn widget(&self) -> &DrawingArea {
        &self.drawing_area
    }

    /// Set the tensor data to visualize
    /// Automatically downsamples intelligently based on display resolution
    pub fn set_data(&self, values: Vec<f32>, original_shape: &[u64]) {
        let heatmap_data = prepare_heatmap_data(values, original_shape);

        if let Some(data) = &heatmap_data {
            // Calculate optimal size for the drawing area (add space for legend)
            let aspect_ratio = data.width as f64 / data.height as f64;
            let legend_space = 40.0;

            let (content_width, content_height) = if aspect_ratio > 1.0 {
                // Wider than tall
                let w = MAX_DISPLAY_WIDTH.min(data.width as f64 * MIN_CELL_SIZE);
                let h = (w / aspect_ratio).min(MAX_DISPLAY_HEIGHT - legend_space);
                (w as i32, h as i32 + legend_space as i32)
            } else {
                // Taller than wide or square
                let h = (MAX_DISPLAY_HEIGHT - legend_space).min(data.height as f64 * MIN_CELL_SIZE);
                let w = (h * aspect_ratio).min(MAX_DISPLAY_WIDTH);
                (w as i32, h as i32 + legend_space as i32)
            };

            self.drawing_area.set_content_width(content_width.max(200).min(MAX_DISPLAY_WIDTH as i32));
            self.drawing_area.set_content_height(content_height.max(240).min((MAX_DISPLAY_HEIGHT + 40.0) as i32));
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

    /// Toggle between heatmap and histogram display modes
    pub fn toggle_display_mode(&self) {
        let mut mode = self.display_mode.borrow_mut();
        *mode = match *mode {
            DisplayMode::Heatmap => DisplayMode::Histogram,
            DisplayMode::Histogram => DisplayMode::Heatmap,
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
}

/// Prepare and downsample tensor data for heatmap visualization
fn prepare_heatmap_data(values: Vec<f32>, shape: &[u64]) -> Option<HeatmapData> {
    if values.is_empty() || shape.is_empty() {
        return None;
    }

    // Determine 2D layout from shape
    let (width, height) = match shape.len() {
        1 => {
            // 1D tensor - display as a horizontal strip or wrap into 2D
            let total = shape[0] as usize;
            if total <= MAX_HEATMAP_DIMENSION {
                (total, 1)
            } else {
                // Wrap into a square-ish shape
                let side = (total as f64).sqrt().ceil() as usize;
                (side, (total + side - 1) / side)
            }
        },
        2 => (shape[1] as usize, shape[0] as usize),
        3 => {
            // For 3D tensors (e.g., [batch, height, width]), flatten batch dimension
            (shape[2] as usize, (shape[0] * shape[1]) as usize)
        },
        _ => {
            // For higher dimensional tensors, flatten all but last two dimensions
            let w = shape[shape.len() - 1] as usize;
            let h: usize = shape[..shape.len() - 1].iter().map(|&d| d as usize).product();
            (w, h)
        }
    };

    // Downsample if necessary
    let (final_width, final_height, downsampled_values) =
        if width > MAX_HEATMAP_DIMENSION || height > MAX_HEATMAP_DIMENSION {
            downsample_data(&values, width, height)
        } else {
            (width, height, values)
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
        width: final_width,
        height: final_height,
        min_value,
        max_value,
    })
}

/// Downsample a 2D array to fit within MAX_HEATMAP_DIMENSION
fn downsample_data(values: &[f32], width: usize, height: usize) -> (usize, usize, Vec<f32>) {
    let scale_w = (width as f64 / MAX_HEATMAP_DIMENSION as f64).ceil();
    let scale_h = (height as f64 / MAX_HEATMAP_DIMENSION as f64).ceil();
    let scale = scale_w.max(scale_h) as usize;

    let new_width = (width + scale - 1) / scale;
    let new_height = (height + scale - 1) / scale;

    let mut downsampled = vec![0.0f32; new_width * new_height];

    for out_y in 0..new_height {
        for out_x in 0..new_width {
            let in_x_start = out_x * scale;
            let in_y_start = out_y * scale;
            let in_x_end = (in_x_start + scale).min(width);
            let in_y_end = (in_y_start + scale).min(height);

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
            downsampled[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    (new_width, new_height, downsampled)
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
                cell_width,
                cell_height,
            );
            cr.fill().unwrap();
        }
    }

    // Draw subtle grid lines if cells are large enough
    if cell_width >= 4.0 && cell_height >= 4.0 {
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.1);
        cr.set_line_width(0.5);

        for x in 0..=data.width {
            let px = x as f64 * cell_width;
            cr.move_to(px, 0.0);
            cr.line_to(px, heatmap_height);
        }

        for y in 0..=data.height {
            let py = y as f64 * cell_height;
            cr.move_to(0.0, py);
            cr.line_to(width, py);
        }

        cr.stroke().unwrap();
    }

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
            segment_width.ceil(),
            bar_height,
        );
        cr.fill().unwrap();
    }

    // Draw border around gradient bar
    cr.set_source_rgba(0.5, 0.5, 0.5, 0.5);
    cr.set_line_width(1.0);
    cr.rectangle(margin, bar_y, bar_width, bar_height);
    cr.stroke().unwrap();

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

        // Color bars using memory visualizer palette (green-blue gradient)
        let t = i as f64 / (num_bins - 1) as f64;
        let (r, g, b) = memory_viz_color_gradient(t);

        cr.set_source_rgba(r, g, b, 0.85);
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
fn memory_viz_color_gradient(t: f64) -> (f64, f64, f64) {
    let t = t.clamp(0.0, 1.0);

    // Gradient from green (low values) to blue (high values)
    // Matches the color scheme used in tensor_viewer.rs for memory visualization
    // Green (muted): RGB(0.37, 0.62, 0.31) -> Green (bright): RGB(0.46, 0.78, 0.39)
    // Blue (muted): RGB(0.20, 0.47, 0.80) -> Blue (bright): RGB(0.25, 0.59, 0.95)

    if t < 0.5 {
        // Green range (low to medium values)
        let local_t = t * 2.0; // Map 0-0.5 to 0-1
        let r = 0.37 + (0.46 - 0.37) * local_t;
        let g = 0.62 + (0.78 - 0.62) * local_t;
        let b = 0.31 + (0.39 - 0.31) * local_t;
        (r, g, b)
    } else {
        // Transition to blue range (medium to high values)
        let local_t = (t - 0.5) * 2.0; // Map 0.5-1.0 to 0-1
        let r = 0.46 + (0.25 - 0.46) * local_t;
        let g = 0.78 + (0.59 - 0.78) * local_t;
        let b = 0.39 + (0.95 - 0.39) * local_t;
        (r, g, b)
    }
}
