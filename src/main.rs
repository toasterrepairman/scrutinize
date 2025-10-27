use gtk::prelude::*;
use gtk::gio;

mod gguf_parser;
mod ui;
mod tokenizer;
mod tensor_viewer;
mod token_display;
mod heatmap_widget;
mod optimized_tokenizer;

use ui::window::GGUFWindow;

const APP_ID: &str = "com.github.scrutinize";

fn main() -> glib::ExitCode {
    // Create the application
    let app = adw::Application::builder()
        .application_id(APP_ID)
        .flags(gio::ApplicationFlags::HANDLES_OPEN)
        .build();

    app.connect_startup(|_| {
        adw::init().expect("Failed to initialize libadwaita");

        // Load CSS
        let provider = gtk::CssProvider::new();
        provider.load_from_resource("/com/github/scrutinize/style.css");

        gtk::style_context_add_provider_for_display(
            &gtk::gdk::Display::default().expect("Could not connect to a display."),
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    });

    app.connect_activate(build_ui);
    app.connect_open(open_files);

    app.run()
}

fn build_ui(app: &adw::Application) {
    let window = GGUFWindow::new(app);
    window.present();
}

fn open_files(app: &adw::Application, files: &[gio::File], _hint: &str) {
    let window = GGUFWindow::new(app);

    if let Some(file) = files.first() {
        if let Some(path) = file.path() {
            window.load_file(&path);
        }
    }

    window.present();
}
