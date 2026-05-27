use gtk::gdk::{MemoryFormat, MemoryTexture};
use gtk::glib;
use gtk::gsk::{ScalingFilter, TextureScaleNode};
use gtk::graphene;
use gtk::prelude::*;
use gtk::subclass::prelude::*;
use gtk::{
    Adjustment, Box as GtkBox, DrawingArea, EventControllerScroll, EventControllerScrollFlags,
    GestureZoom, Label, Orientation, Scrollbar, Widget,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

mod imp {
    use super::*;
    use glib::subclass::InitializingObject;

    #[derive(Default)]
    pub struct TensorCanvas {
        pub texture: RefCell<Option<MemoryTexture>>,
        pub tex_width: Cell<i32>,
        pub tex_height: Cell<i32>,
        pub zoom: Cell<f64>,
        pub min_zoom: Cell<f64>,
        pub fit_zoom: Cell<f64>,
        pub hadj: RefCell<Option<Adjustment>>,
        pub vadj: RefCell<Option<Adjustment>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for TensorCanvas {
        const NAME: &'static str = "ScrutinizeTensorCanvas";
        type Type = super::TensorCanvas;
        type ParentType = Widget;

        fn class_init(klass: &mut Self::Class) {
            klass.set_css_name("tensor-canvas");
        }

        fn instance_init(_obj: &InitializingObject<Self>) {}
    }

    impl ObjectImpl for TensorCanvas {
        fn constructed(&self) {
            self.parent_constructed();
            let obj = self.obj();
            obj.set_overflow(gtk::Overflow::Hidden);
        }
    }

    impl WidgetImpl for TensorCanvas {
        fn request_mode(&self) -> gtk::SizeRequestMode {
            gtk::SizeRequestMode::ConstantSize
        }

        fn measure(&self, orientation: gtk::Orientation, _for_size: i32) -> (i32, i32, i32, i32) {
            let z = self.zoom.get();
            let size = match orientation {
                gtk::Orientation::Horizontal => {
                    let raw = self.tex_width.get() as f64;
                    (raw * z).ceil() as i32
                }
                _ => {
                    let raw = self.tex_height.get() as f64;
                    (raw * z).ceil() as i32
                }
            };
            (0, size.max(1), -1, -1)
        }

        fn snapshot(&self, snapshot: &gtk::Snapshot) {
            let texture = match self.texture.borrow().as_ref() {
                Some(t) => t.clone(),
                None => return,
            };

            let z = self.zoom.get() as f32;
            let tw = texture.width() as f32 * z;
            let th = texture.height() as f32 * z;

            let sx = self.hadj.borrow().as_ref().map(|a| a.value() as f32).unwrap_or(0.0);
            let sy = self.vadj.borrow().as_ref().map(|a| a.value() as f32).unwrap_or(0.0);

            snapshot.save();
            snapshot.translate(&graphene::Point::new(-sx, -sy));

            let node = TextureScaleNode::new(
                &texture,
                &graphene::Rect::new(0.0, 0.0, tw, th),
                ScalingFilter::Nearest,
            );
            snapshot.append_node(&node);
            snapshot.restore();
        }

        fn size_allocate(&self, _width: i32, _height: i32, _baseline: i32) {
            let obj = self.obj();
            obj.update_adjustments();
        }
    }
}

glib::wrapper! {
    pub struct TensorCanvas(ObjectSubclass<imp::TensorCanvas>)
        @extends Widget,
        @implements gtk::Accessible, gtk::Buildable, gtk::ConstraintTarget;
}

impl TensorCanvas {
    pub fn new() -> Self {
        glib::Object::new::<Self>()
    }

    pub fn set_texture(&self, texture: MemoryTexture) {
        let w = texture.width();
        let h = texture.height();
        self.imp().tex_width.set(w);
        self.imp().tex_height.set(h);
        self.imp().texture.replace(Some(texture));
        self.imp().zoom.set(1.0);
        self.queue_resize();
    }

    pub fn set_zoom(&self, zoom: f64) {
        let old = self.imp().zoom.get();
        self.imp().zoom.set(zoom);
        self.update_adjustments();
        if (zoom - old).abs() > 0.001 {
            self.queue_draw();
        }
    }

    pub fn zoom(&self) -> f64 {
        self.imp().zoom.get()
    }

    pub fn set_fit_zoom(&self, z: f64) {
        self.imp().fit_zoom.set(z);
    }

    pub fn fit_zoom(&self) -> f64 {
        self.imp().fit_zoom.get()
    }

    pub fn set_min_zoom(&self, z: f64) {
        self.imp().min_zoom.set(z);
    }

    pub fn set_adjustments(&self, hadj: Adjustment, vadj: Adjustment) {
        let weak = self.downgrade();
        hadj.connect_value_changed(move |_| {
            if let Some(c) = weak.upgrade() {
                c.queue_draw();
            }
        });
        let weak = self.downgrade();
        vadj.connect_value_changed(move |_| {
            if let Some(c) = weak.upgrade() {
                c.queue_draw();
            }
        });
        self.imp().hadj.replace(Some(hadj));
        self.imp().vadj.replace(Some(vadj));
    }

    pub fn update_adjustments(&self) {
        let z = self.imp().zoom.get();
        let tw = self.imp().tex_width.get() as f64;
        let th = self.imp().tex_height.get() as f64;
        let zoomed_w = (tw * z).ceil();
        let zoomed_h = (th * z).ceil();

        let alloc = self.allocation();
        let alloc_w = alloc.width() as f64;
        let alloc_h = alloc.height() as f64;

        if let Some(hadj) = self.imp().hadj.borrow().as_ref() {
            let page = alloc_w.min(zoomed_w);
            hadj.configure(hadj.value().min((zoomed_w - page).max(0.0)), 0.0, zoomed_w, page * 0.1, page * 0.9, page);
        }
        if let Some(vadj) = self.imp().vadj.borrow().as_ref() {
            let page = alloc_h.min(zoomed_h);
            vadj.configure(vadj.value().min((zoomed_h - page).max(0.0)), 0.0, zoomed_h, page * 0.1, page * 0.9, page);
        }
    }

    pub fn texture_dimensions(&self) -> (i32, i32) {
        (self.imp().tex_width.get(), self.imp().tex_height.get())
    }

    pub fn clear(&self) {
        self.imp().texture.replace(None);
        self.imp().tex_width.set(0);
        self.imp().tex_height.set(0);
        self.imp().zoom.set(1.0);
        self.queue_resize();
    }
}

impl Default for TensorCanvas {
    fn default() -> Self {
        Self::new()
    }
}

fn color_at(t: f64) -> (f64, f64, f64) {
    let t = t.clamp(0.0, 1.0);
    if t < 0.33 {
        let s = t / 0.33;
        (0.37 + 0.09 * s, 0.62 + 0.16 * s, 0.31 + 0.08 * s)
    } else if t < 0.66 {
        let s = (t - 0.33) / 0.33;
        (0.46 - 0.11 * s, 0.78 - 0.08 * s, 0.39 + 0.31 * s)
    } else {
        let s = (t - 0.66) / 0.34;
        (0.35 - 0.10 * s, 0.70 - 0.11 * s, 0.70 + 0.25 * s)
    }
}

pub fn tensor_to_rgba(values: &[f32], min: f32, max: f32) -> (Vec<u8>, usize) {
    let range = (max - min).abs().max(0.0001);
    let len = values.len();
    let mut pixels = Vec::with_capacity(len * 4);
    for &v in values {
        let t = ((v - min) / range).clamp(0.0, 1.0) as f64;
        let (r, g, b) = color_at(t);
        pixels.push((r * 255.0) as u8);
        pixels.push((g * 255.0) as u8);
        pixels.push((b * 255.0) as u8);
        pixels.push(255u8);
    }
    (pixels, len)
}

#[derive(Clone)]
pub struct ZoomableView {
    container: GtkBox,
    canvas: TensorCanvas,
    hadj: Adjustment,
    vadj: Adjustment,
    legend_bar: DrawingArea,
    legend_min: Label,
    legend_max: Label,
    zoom_label: Label,
    width: Cell<i32>,
    height: Cell<i32>,
    zoom: Cell<f64>,
    min_val: Cell<f64>,
    max_val: Cell<f64>,
    gesture_zoom_start: Cell<f64>,
}

impl ZoomableView {
    pub fn new() -> Self {
        let container = GtkBox::new(Orientation::Vertical, 4);
        container.add_css_class("heatmap-container");

        let canvas = TensorCanvas::new();

        let hadj = Adjustment::new(0.0, 0.0, 1.0, 0.1, 1.0, 1.0);
        let vadj = Adjustment::new(0.0, 0.0, 1.0, 0.1, 1.0, 1.0);
        canvas.set_adjustments(hadj.clone(), vadj.clone());

        let hscroll = Scrollbar::new(Orientation::Horizontal, Some(&hadj));
        hscroll.set_hexpand(true);

        let vscroll = Scrollbar::new(Orientation::Vertical, Some(&vadj));
        vscroll.set_vexpand(true);

        let canvas_frame = gtk::Frame::new(None);
        canvas_frame.set_child(Some(&canvas));
        canvas_frame.add_css_class("card");
        canvas_frame.set_height_request(380);

        let grid = GtkBox::new(Orientation::Vertical, 0);
        grid.set_hexpand(true);
        grid.append(&canvas_frame);
        grid.append(&hscroll);

        let outer = GtkBox::new(Orientation::Horizontal, 0);
        outer.set_hexpand(true);
        outer.append(&grid);
        outer.append(&vscroll);

        let scroll_ctrl = EventControllerScroll::builder()
            .flags(EventControllerScrollFlags::VERTICAL | EventControllerScrollFlags::DISCRETE)
            .build();

        let gesture_zoom = GestureZoom::new();

        let legend_bar = DrawingArea::new();
        legend_bar.set_content_width(200);
        legend_bar.set_content_height(12);
        legend_bar.add_css_class("heatmap-legend");
        legend_bar.set_hexpand(true);
        legend_bar.set_draw_func(|_, cr, w, h| {
            use gtk::cairo;
            let grad = cairo::LinearGradient::new(0.0, 0.0, w as f64, 0.0);
            for i in 0..=20 {
                let t = i as f64 / 20.0;
                let (r, g, b) = color_at(t);
                grad.add_color_stop_rgb(t, r, g, b);
            }
            cr.set_source(&grad).unwrap();
            cr.set_fill_rule(cairo::FillRule::Winding);
            let r = 4.0;
            cr.move_to(r, 0.0);
            cr.line_to(w as f64 - r, 0.0);
            cr.arc(
                w as f64 - r,
                r,
                r,
                -std::f64::consts::FRAC_PI_2,
                std::f64::consts::FRAC_PI_2,
            );
            cr.line_to(r, h as f64);
            cr.arc(
                r,
                r,
                r,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::FRAC_PI_2 * 3.0,
            );
            cr.close_path();
            cr.fill().unwrap();
        });

        let legend_min = Label::new(Some("0.0"));
        legend_min.add_css_class("dim-label");
        legend_min.add_css_class("caption");
        legend_min.set_halign(gtk::Align::Start);

        let legend_max = Label::new(Some("1.0"));
        legend_max.add_css_class("dim-label");
        legend_max.add_css_class("caption");
        legend_max.set_halign(gtk::Align::End);

        let legend_labels = GtkBox::new(Orientation::Horizontal, 0);
        legend_labels.set_hexpand(true);
        legend_labels.append(&legend_min);
        legend_labels.append(&legend_max);
        let legend_box = GtkBox::new(Orientation::Vertical, 2);
        legend_box.append(&legend_bar);
        legend_box.append(&legend_labels);

        let zoom_label = Label::new(Some("100%"));
        zoom_label.add_css_class("zoom-label");
        zoom_label.add_css_class("monospace");
        zoom_label.set_visible(false);

        let info_bar = GtkBox::new(Orientation::Horizontal, 8);
        info_bar.set_margin_start(4);
        info_bar.set_margin_end(4);
        info_bar.append(&zoom_label);

        let main_box = GtkBox::new(Orientation::Vertical, 6);
        main_box.append(&outer);
        main_box.append(&legend_box);
        main_box.append(&info_bar);

        container.append(&main_box);

        let z_clone_ref = Rc::new(RefCell::new(None::<ZoomableView>));

        let z_clone = z_clone_ref.clone();
        scroll_ctrl.connect_scroll(move |_, _, dy| {
            if let Some(z) = z_clone.borrow().as_ref() {
                z.adjust_zoom(if dy < 0.0 { 1.2 } else { 1.0 / 1.2 });
            }
            glib::Propagation::Proceed
        });
        canvas.add_controller(scroll_ctrl);

        let z_clone = z_clone_ref.clone();
        gesture_zoom.connect_begin(move |_, _| {
            if let Some(z) = z_clone.borrow().as_ref() {
                z.gesture_zoom_start.set(z.zoom.get());
            }
        });

        let z_clone = z_clone_ref.clone();
        gesture_zoom.connect_scale_changed(move |_, scale| {
            if let Some(z) = z_clone.borrow().as_ref() {
                let start = z.gesture_zoom_start.get();
                if start <= 0.0 {
                    z.gesture_zoom_start.set(z.zoom.get());
                    return;
                }
                let new_zoom = start * scale;
                let clamped = new_zoom.clamp(0.1, 100.0);
                z.zoom.set(clamped);
                z.canvas.set_zoom(clamped);
                z.update_zoom_label();
            }
        });
        canvas.add_controller(gesture_zoom);

        let slf = ZoomableView {
            container: container.clone(),
            canvas: canvas.clone(),
            hadj,
            vadj,
            legend_bar: legend_bar.clone(),
            legend_min: legend_min.clone(),
            legend_max: legend_max.clone(),
            zoom_label: zoom_label.clone(),
            width: Cell::new(0),
            height: Cell::new(0),
            zoom: Cell::new(1.0),
            min_val: Cell::new(0.0),
            max_val: Cell::new(1.0),
            gesture_zoom_start: Cell::new(1.0),
        };

        *z_clone_ref.borrow_mut() = Some(slf.clone());

        slf
    }

    fn adjust_zoom(&self, factor: f64) {
        let new = (self.zoom.get() * factor).clamp(0.1, 100.0);
        self.zoom.set(new);
        self.canvas.set_zoom(new);
        self.update_zoom_label();
    }

    fn update_zoom_label(&self) {
        let z = self.zoom.get();
        self.zoom_label.set_text(&format!("{:.0}%", z * 100.0));
        self.zoom_label.set_visible(true);
    }

    pub fn widget(&self) -> &gtk::Widget {
        self.container.upcast_ref()
    }

    pub fn canvas(&self) -> &TensorCanvas {
        &self.canvas
    }

    pub fn set_tensor_data(&self, values: &[f32], width: i32, height: i32, min_val: f32, max_val: f32) {
        let actual_len = values.len() as i64;
        let declared = (width as i64) * (height as i64);
        let (tex_w, tex_h) = if actual_len != declared && width > 0 {
            let w = width as i64;
            let h = (actual_len + w - 1) / w;
            (width, h as i32)
        } else {
            (width, height)
        };

        self.width.set(tex_w);
        self.height.set(tex_h);
        self.min_val.set(min_val as f64);
        self.max_val.set(max_val as f64);

        let (pixels, _len) = tensor_to_rgba(values, min_val, max_val);
        let stride = (tex_w as usize) * 4;
        let expected = stride * (tex_h as usize);
        let pixels = if pixels.len() < expected {
            let mut p = pixels;
            p.resize(expected, 0);
            p
        } else {
            pixels
        };

        let bytes = glib::Bytes::from_owned(pixels);
        let texture = MemoryTexture::new(
            tex_w,
            tex_h,
            MemoryFormat::R8g8b8a8,
            &bytes,
            stride,
        );

        self.canvas.set_texture(texture);

        let initial_zoom = 1.0_f64.min(560.0 / (tex_w as f64).max(1.0)).min(400.0 / (tex_h as f64).max(1.0));
        self.zoom.set(initial_zoom);
        self.canvas.set_zoom(initial_zoom);
        self.canvas.set_fit_zoom(initial_zoom);
        self.canvas.set_min_zoom(initial_zoom * 0.1);
        self.hadj.set_value(0.0);
        self.vadj.set_value(0.0);
        self.update_zoom_label();
    }

    pub fn set_legend(&self, min: f32, max: f32) {
        self.min_val.set(min as f64);
        self.max_val.set(max as f64);
        if min.is_finite() && max.is_finite() {
            self.legend_min.set_text(&format!("{:.4}", min));
            self.legend_max.set_text(&format!("{:.4}", max));
            self.legend_bar.set_visible(true);
            self.legend_min.set_visible(true);
            self.legend_max.set_visible(true);
        } else {
            self.legend_bar.set_visible(false);
            self.legend_min.set_visible(false);
            self.legend_max.set_visible(false);
        }
    }

    pub fn zoom_level(&self) -> f64 {
        self.zoom.get()
    }

    pub fn data_dimensions(&self) -> (i32, i32) {
        (self.width.get(), self.height.get())
    }

    pub fn clear(&self) {
        self.canvas.clear();
        self.width.set(0);
        self.height.set(0);
        self.zoom.set(1.0);
        self.legend_bar.set_visible(false);
        self.legend_min.set_visible(false);
        self.legend_max.set_visible(false);
        self.zoom_label.set_visible(false);
    }
}

impl Default for ZoomableView {
    fn default() -> Self {
        Self::new()
    }
}
