use crate::catalog::{
    self, Catalog, Filter, Label, PAGE_SIZE, QueryResult, SPLITS, Sample, Summary,
};
use anyhow::Result;
use eframe::egui::{self, Color32, RichText, TextureHandle};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
};

const AMBER: Color32 = Color32::from_rgb(192, 139, 24);
const INK: Color32 = Color32::from_rgb(38, 49, 43);
const MUTED: Color32 = Color32::from_rgb(114, 127, 112);
const SPLIT_COLORS: [Color32; 3] = [
    Color32::from_rgb(220, 169, 45),
    Color32::from_rgb(78, 145, 136),
    Color32::from_rgb(155, 124, 179),
];
const DEFAULT_QUERY: &str = "SELECT id, source, collected_at, split, width, height, image_path, mask_path\nFROM dataset ORDER BY id;";

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Overview,
    Sources,
    Sql,
}

type JobResult = std::result::Result<String, String>;
type ThumbnailResult = (u64, i64, std::result::Result<egui::ColorImage, String>);

pub struct App {
    db: Catalog,
    summary: Summary,
    tab: Tab,
    filter: Filter,
    offset: usize,
    total: usize,
    samples: Vec<Sample>,
    overlay: bool,
    textures: HashMap<i64, TextureHandle>,
    image_errors: HashMap<i64, String>,
    thumbnail_rx: mpsc::Receiver<ThumbnailResult>,
    thumbnail_tx: mpsc::Sender<ThumbnailResult>,
    generation: Arc<AtomicU64>,
    job: Option<mpsc::Receiver<JobResult>>,
    status: String,
    failed: bool,
    selected_sample: Option<Sample>,
    selected_source: Option<String>,
    location: String,
    group_target: String,
    import_split: String,
    import_group: String,
    source_name: String,
    source_path: String,
    source_date: String,
    source_notes: String,
    sql: String,
    result: Option<QueryResult>,
    schema: String,
    snapshot_path: String,
}

pub fn run(db: Catalog) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("BEE detector · Dataset")
            .with_inner_size([1320.0, 900.0])
            .with_min_inner_size([800.0, 600.0]),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "bee-detector",
        options,
        Box::new(move |cc| {
            style(&cc.egui_ctx);
            Ok(Box::new(App::new(db, &cc.egui_ctx)?))
        }),
    )
    .map_err(|error| anyhow::anyhow!("Cannot open the Linux window: {error}"))
}

fn style(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::light();
    visuals.panel_fill = Color32::from_rgb(247, 248, 244);
    visuals.window_fill = Color32::WHITE;
    visuals.override_text_color = Some(INK);
    visuals.selection.bg_fill = Color32::from_rgb(213, 231, 222);
    visuals.selection.stroke.color = INK;
    ctx.set_visuals(visuals);
    ctx.style_mut(|style| {
        style.spacing.item_spacing = egui::vec2(12.0, 12.0);
        style.spacing.button_padding = egui::vec2(12.0, 8.0);
    });
}

fn bento() -> egui::Frame {
    egui::Frame::new()
        .fill(Color32::WHITE)
        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(225, 230, 220)))
        .corner_radius(14)
        .inner_margin(18)
}

impl App {
    fn new(db: Catalog, ctx: &egui::Context) -> Result<Self> {
        let (thumbnail_tx, thumbnail_rx) = mpsc::channel();
        let snapshot_path = db
            .path
            .with_file_name("catalog.snapshot.sql")
            .display()
            .to_string();
        let mut app = Self {
            summary: db.summary()?,
            schema: db.schema()?,
            db,
            tab: Tab::Overview,
            filter: Filter::default(),
            offset: 0,
            total: 0,
            samples: Vec::new(),
            overlay: true,
            textures: HashMap::new(),
            image_errors: HashMap::new(),
            thumbnail_rx,
            thumbnail_tx,
            generation: Arc::new(AtomicU64::new(0)),
            job: None,
            status: String::new(),
            failed: false,
            selected_sample: None,
            selected_source: None,
            location: String::new(),
            group_target: "train".into(),
            import_split: "train".into(),
            import_group: String::new(),
            source_name: String::new(),
            source_path: String::new(),
            source_date: String::new(),
            source_notes: String::new(),
            sql: DEFAULT_QUERY.into(),
            result: None,
            snapshot_path,
        };
        app.reload(ctx)?;
        Ok(app)
    }

    fn reload(&mut self, ctx: &egui::Context) -> Result<()> {
        self.summary = self.db.summary()?;
        self.schema = self.db.schema()?;
        self.load_page(ctx)
    }

    fn load_page(&mut self, ctx: &egui::Context) -> Result<()> {
        (self.total, self.samples) = self.db.page(&self.filter, self.offset)?;
        self.textures.clear();
        self.image_errors.clear();
        let generation = self.generation.fetch_add(1, Ordering::Relaxed) + 1;
        let token = self.generation.clone();
        let samples = self.samples.clone();
        let labels = self.summary.labels.clone();
        let overlay = self.overlay;
        let sender = self.thumbnail_tx.clone();
        let ctx = ctx.clone();
        thread::spawn(move || {
            for sample in samples {
                if token.load(Ordering::Relaxed) != generation {
                    break;
                }
                let result = thumbnail(&sample, overlay, &labels).map_err(|e| format!("{e:#}"));
                if sender.send((generation, sample.id, result)).is_err() {
                    break;
                }
                ctx.request_repaint();
            }
        });
        Ok(())
    }

    fn background(
        &mut self,
        ctx: &egui::Context,
        task: impl FnOnce(&mut Catalog) -> Result<String> + Send + 'static,
    ) {
        if self.job.is_some() {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        let path = self.db.path.clone();
        let ctx = ctx.clone();
        self.job = Some(receiver);
        self.status = "Working…".into();
        self.failed = false;
        thread::spawn(move || {
            let result = Catalog::open(&path)
                .and_then(|mut db| task(&mut db))
                .map_err(|e| format!("{e:#}"));
            let _ = sender.send(result);
            ctx.request_repaint();
        });
    }

    fn report(&mut self, result: Result<()>) {
        if let Err(error) = result {
            self.status = format!("{error:#}");
            self.failed = true;
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok((generation, id, result)) = self.thumbnail_rx.try_recv() {
            if generation != self.generation.load(Ordering::Relaxed) {
                continue;
            }
            match result {
                Ok(image) => {
                    self.textures.insert(
                        id,
                        ctx.load_texture(
                            format!("{generation}-{id}"),
                            image,
                            egui::TextureOptions::LINEAR,
                        ),
                    );
                }
                Err(error) => {
                    self.image_errors.insert(id, error);
                }
            }
        }
        let result = self.job.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(result) = result {
            self.job = None;
            match result {
                Ok(message) => {
                    self.status = message;
                    self.failed = false;
                }
                Err(error) => {
                    self.status = error;
                    self.failed = true;
                }
            }
            self.offset = 0;
            let result = self.reload(ctx);
            self.report(result);
            if self.tab == Tab::Sql {
                self.result = self.db.query(&self.sql).ok();
            }
        }
    }

    fn draw(&mut self, ctx: &egui::Context) {
        self.poll(ctx);
        egui::TopBottomPanel::top("header")
            .frame(egui::Frame::new().fill(Color32::WHITE).inner_margin(20))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("BEE").size(26.0).strong().color(AMBER));
                    ui.label(RichText::new("detector").size(26.0).strong());
                    ui.label(RichText::new("/  dataset").color(MUTED));
                    ui.add_space(30.0);
                    for (tab, title) in [
                        (Tab::Overview, "Overview"),
                        (Tab::Sources, "Sources"),
                        (Tab::Sql, "SQL catalog"),
                    ] {
                        ui.selectable_value(&mut self.tab, tab, title);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(self.job.is_none(), egui::Button::new("Refresh"))
                            .clicked()
                        {
                            let result = self.reload(ctx);
                            self.report(result);
                        }
                        ui.label(
                            RichText::new("LOCAL · LINUX · RUST")
                                .size(10.0)
                                .color(MUTED),
                        );
                    });
                });
            });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if self.job.is_some() {
                    ui.spinner();
                }
                if !self.status.is_empty() {
                    ui.label(RichText::new(&self.status).color(if self.failed {
                        Color32::from_rgb(162, 62, 44)
                    } else {
                        MUTED
                    }));
                } else {
                    ui.label(
                        RichText::new(self.db.path.display().to_string())
                            .small()
                            .color(MUTED),
                    );
                }
            });
        });
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .inner_margin(24)
                    .fill(Color32::from_rgb(247, 248, 244)),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("main-scroll")
                    .show(ui, |ui| match self.tab {
                        Tab::Overview => self.overview(ui, ctx),
                        Tab::Sources => self.sources(ui, ctx),
                        Tab::Sql => self.sql_view(ui, ctx),
                    });
            });
        self.source_window(ctx);
        self.sample_window(ctx);
    }

    fn overview(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label(
            RichText::new("BETTER DATA. HEALTHIER HIVES.")
                .small()
                .color(MUTED),
        );
        ui.heading(RichText::new("Your hive data, in one place.").size(30.0));
        ui.label("Browse collections and segmentation masks. SQL defines the splits; folders can live anywhere.");
        ui.add_space(8.0);
        let total: usize = self.summary.splits.iter().sum();
        let width = ((ui.available_width() - 36.0) / 4.0).max(150.0);
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            for index in 0..4 {
                let (name, count, note, color) = if index == 0 {
                    (
                        "All images",
                        total,
                        format!(
                            "{} sources · {} groups",
                            self.summary.sources.len(),
                            self.summary.groups
                        ),
                        INK,
                    )
                } else {
                    let count = self.summary.splits[index - 1];
                    (
                        SPLITS[index - 1],
                        count,
                        format!(
                            "{:.1}% of the dataset",
                            if total > 0 {
                                count as f64 / total as f64 * 100.0
                            } else {
                                0.0
                            }
                        ),
                        SPLIT_COLORS[index - 1],
                    )
                };
                let card = bento().show(ui, |ui| {
                    ui.set_min_width(width - 36.0);
                    ui.label(RichText::new(name).color(color));
                    ui.label(RichText::new(count.to_string()).size(36.0).strong());
                    ui.label(RichText::new(note).small().color(MUTED));
                });
                if ui
                    .interact(
                        card.response.rect,
                        ui.id().with(("split-card", index)),
                        egui::Sense::click(),
                    )
                    .clicked()
                {
                    self.filter.split = if index == 0 {
                        String::new()
                    } else {
                        SPLITS[index - 1].into()
                    };
                    self.offset = 0;
                    changed = true;
                }
            }
        });
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("split")
                .selected_text(if self.filter.split.is_empty() {
                    "All splits"
                } else {
                    &self.filter.split
                })
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(&mut self.filter.split, String::new(), "All splits")
                        .changed();
                    for split in SPLITS {
                        changed |= ui
                            .selectable_value(&mut self.filter.split, split.into(), split)
                            .changed();
                    }
                });
            egui::ComboBox::from_id_salt("source")
                .width(220.0)
                .selected_text(if self.filter.source.is_empty() {
                    "All sources"
                } else {
                    &self.filter.source
                })
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(&mut self.filter.source, String::new(), "All sources")
                        .changed();
                    for source in &self.summary.sources {
                        changed |= ui
                            .selectable_value(
                                &mut self.filter.source,
                                source.name.clone(),
                                &source.name,
                            )
                            .changed();
                    }
                });
            let edit = ui.add(
                egui::TextEdit::singleline(&mut self.filter.search)
                    .hint_text("Filename, source, or group")
                    .desired_width(220.0),
            );
            if ui.button("Search").clicked()
                || (edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
            {
                changed = true;
            }
            if ui.checkbox(&mut self.overlay, "Mask overlay").changed() {
                changed = true;
            }
        });
        ui.collapsing("Label coverage", |ui| {
            ui.label(
                RichText::new("Images containing each label; counts overlap.")
                    .small()
                    .color(MUTED),
            );
            ui.horizontal_wrapped(|ui| {
                changed |= ui
                    .selectable_value(&mut self.filter.label, None, "All labels")
                    .changed();
                for label in &self.summary.labels {
                    let text = RichText::new(format!("{}  {}", label.name, label.images)).color(
                        Color32::from_rgb(label.color[0], label.color[1], label.color[2]),
                    );
                    changed |= ui
                        .selectable_value(&mut self.filter.label, Some(label.id), text)
                        .changed();
                }
            });
        });
        if changed {
            self.offset = 0;
            let result = self.load_page(ctx);
            self.report(result);
        }
        ui.add_space(8.0);
        ui.label(RichText::new(format!("Samples · {}", self.total)).strong());
        if self.samples.is_empty() {
            bento().show(ui,|ui| {
                ui.heading(if self.summary.sources.is_empty(){"Start with a collection"}else{"No samples match this view"});
                ui.label("Register a folder in Sources, then index its image/mask pairs. The repository contains no example data.");
                if ui.button("Open sources").clicked(){self.tab=Tab::Sources;}
            });
        }
        let columns = (ui.available_width() / 245.0).floor().max(1.0) as usize;
        let items = self.samples.clone();
        for chunk in items.chunks(columns) {
            ui.columns(columns, |columns| {
                for (column, sample) in columns.iter_mut().zip(chunk) {
                    self.sample_card(column, sample);
                }
            });
        }
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "{}–{} of {}",
                    if self.total == 0 { 0 } else { self.offset + 1 },
                    (self.offset + PAGE_SIZE).min(self.total),
                    self.total
                ))
                .color(MUTED),
            );
            if ui
                .add_enabled(self.offset > 0, egui::Button::new("Previous"))
                .clicked()
            {
                self.offset = self.offset.saturating_sub(PAGE_SIZE);
                let result = self.load_page(ctx);
                self.report(result);
            }
            if ui
                .add_enabled(
                    self.offset + PAGE_SIZE < self.total,
                    egui::Button::new("Next"),
                )
                .clicked()
            {
                self.offset += PAGE_SIZE;
                let result = self.load_page(ctx);
                self.report(result);
            }
        });
    }

    fn sample_card(&mut self, ui: &mut egui::Ui, sample: &Sample) {
        bento().show(ui, |ui| {
            let size = egui::vec2(ui.available_width(), 150.0);
            let clicked = if let Some(texture) = self.textures.get(&sample.id) {
                ui.add(
                    egui::Image::new(texture)
                        .fit_to_exact_size(size)
                        .sense(egui::Sense::click()),
                )
                .clicked()
            } else {
                let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
                ui.painter()
                    .rect_filled(rect, 8.0, Color32::from_rgb(241, 243, 235));
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    if self.image_errors.contains_key(&sample.id) {
                        "File unavailable"
                    } else {
                        "Loading…"
                    },
                    egui::FontId::proportional(13.0),
                    MUTED,
                );
                response.clicked()
            };
            let name = Path::new(&sample.image)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            ui.add(egui::Label::new(RichText::new(name).strong()).truncate())
                .on_hover_text(&sample.image);
            ui.label(
                RichText::new(format!(
                    "{} × {} px · {}",
                    sample.width, sample.height, sample.split
                ))
                .small()
                .color(MUTED),
            );
            ui.label(RichText::new(&sample.collected).small().color(MUTED));
            if let Some(error) = self.image_errors.get(&sample.id) {
                ui.label(RichText::new(error).small().color(Color32::DARK_RED));
            }
            if clicked || ui.small_button("Inspect").clicked() {
                self.selected_sample = Some(sample.clone());
                self.group_target = sample.split.clone();
            }
        });
        ui.add_space(12.0);
    }

    fn sources(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Data sources");
        ui.label("A source is a named collection folder. Collection dates and file resolutions are stored in SQL.");
        ui.collapsing("Register a source", |ui| {
            egui::Grid::new("register").num_columns(2).show(ui, |ui| {
                ui.label("Collection name");
                ui.add(
                    egui::TextEdit::singleline(&mut self.source_name)
                        .hint_text("hive-07__2026-06-12__camera-a"),
                );
                ui.end_row();
                ui.label("Collected on");
                ui.add(egui::TextEdit::singleline(&mut self.source_date).hint_text("YYYY-MM-DD"));
                ui.end_row();
                ui.label("Folder");
                ui.add(
                    egui::TextEdit::singleline(&mut self.source_path)
                        .hint_text("/path/to/collection"),
                );
                ui.end_row();
                ui.label("Notes / provenance");
                ui.text_edit_singleline(&mut self.source_notes);
                ui.end_row();
            });
            if ui
                .add_enabled(self.job.is_none(), egui::Button::new("Register folder"))
                .clicked()
            {
                let (name, path, date, notes) = (
                    self.source_name.clone(),
                    PathBuf::from(&self.source_path),
                    self.source_date.clone(),
                    self.source_notes.clone(),
                );
                self.background(ctx, move |db| {
                    db.add_source(&name, &path, &date, &notes)?;
                    Ok(format!("Registered {name}"))
                });
            }
        });
        for source in self.summary.sources.clone() {
            bento().show(ui, |ui| {
                ui.label(RichText::new(&source.name).size(17.0).strong());
                ui.label(
                    RichText::new(format!(
                        "Collected {} · {} images · {}",
                        source.collected, source.images, source.resolution
                    ))
                    .color(MUTED),
                );
                ui.monospace(&source.root);
                if !source.notes.is_empty() {
                    ui.label(&source.notes);
                }
                ui.horizontal(|ui| {
                    for (i, split) in SPLITS.iter().enumerate() {
                        ui.label(
                            RichText::new(format!("{split}  {}", source.splits[i]))
                                .color(SPLIT_COLORS[i]),
                        );
                    }
                    if ui.button("Browse").clicked() {
                        self.filter = Filter {
                            source: source.name.clone(),
                            ..Default::default()
                        };
                        self.tab = Tab::Overview;
                        self.offset = 0;
                        let result = self.load_page(ctx);
                        self.report(result);
                    }
                    if ui.button("Manage location / index").clicked() {
                        self.location = self
                            .db
                            .root(&source.name)
                            .unwrap_or_default()
                            .display()
                            .to_string();
                        self.import_group = source.name.clone();
                        self.selected_source = Some(source.name.clone());
                    }
                });
            });
            ui.add_space(12.0);
        }
        if self.summary.sources.is_empty() {
            ui.label("No sources yet. Register an existing collection folder above; its physical layout does not define its split.");
        }
    }

    fn source_window(&mut self, ctx: &egui::Context) {
        let Some(name) = self.selected_source.clone() else {
            return;
        };
        let mut open = true;
        egui::Window::new(format!("Source · {name}")).open(&mut open).default_width(620.0).show(ctx,|ui| {
            ui.label("New source folder location");
            ui.add(egui::TextEdit::singleline(&mut self.location).desired_width(f32::INFINITY));
            ui.label(RichText::new("Repoint checks existing files and hashes. Move folder renames the whole source on the same filesystem.").small().color(MUTED));
            ui.add_enabled_ui(self.job.is_none(),|ui|ui.horizontal(|ui| {
                if ui.button("Repoint catalog").clicked(){let source=name.clone();let path=PathBuf::from(&self.location);self.background(ctx,move|db|{db.relocate(&source,&path)?;Ok("Source location updated".into())});}
                if ui.button("Move folder + update SQL").clicked(){let source=name.clone();let path=PathBuf::from(&self.location);self.background(ctx,move|db|{db.move_source(&source,&path)?;Ok("Source folder moved and catalog updated".into())});}
            }));
            ui.separator();
            ui.heading("Index image/mask pairs");
            ui.label("Automatic pairs use frame.jpg + frame.mask.png, including nested folders. Use register for other mappings.");
            ui.horizontal(|ui|{
                egui::ComboBox::from_id_salt("import-split").selected_text(&self.import_split).show_ui(ui,|ui|{for split in SPLITS{ui.selectable_value(&mut self.import_split,split.into(),split);}});
                ui.label("Group");ui.text_edit_singleline(&mut self.import_group);
            });
            if ui.add_enabled(self.job.is_none(),egui::Button::new("Index source")).clicked(){
                let(source,split,group)=(name.clone(),self.import_split.clone(),self.import_group.clone());
                self.background(ctx,move|db|{let count=db.import(&source,&split,&group)?;Ok(format!("Indexed {count} pairs"))});
            }
        });
        if !open {
            self.selected_source = None;
        }
    }

    fn sample_window(&mut self, ctx: &egui::Context) {
        let Some(sample) = self.selected_sample.clone() else {
            return;
        };
        let mut open = true;
        egui::Window::new(format!("Image {}", sample.id))
            .open(&mut open)
            .default_width(620.0)
            .show(ctx, |ui| {
                if let Some(texture) = self.textures.get(&sample.id) {
                    ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(560.0, 300.0)));
                }
                for (key, value) in [
                    ("Source", sample.source.clone()),
                    ("Collected", sample.collected.clone()),
                    (
                        "Resolution",
                        format!("{} × {} pixels", sample.width, sample.height),
                    ),
                    ("Group", sample.group.clone()),
                    (
                        "Image",
                        sample.root.join(&sample.image).display().to_string(),
                    ),
                    ("Mask", sample.root.join(&sample.mask).display().to_string()),
                ] {
                    ui.label(RichText::new(key).small().color(MUTED));
                    ui.label(value);
                }
                ui.separator();
                ui.label("Assign the whole group to a split. File locations remain independent.");
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("group-split")
                        .selected_text(&self.group_target)
                        .show_ui(ui, |ui| {
                            for split in SPLITS {
                                ui.selectable_value(&mut self.group_target, split.into(), split);
                            }
                        });
                    if ui
                        .add_enabled(self.job.is_none(), egui::Button::new("Update group split"))
                        .clicked()
                    {
                        let (group, split) = (sample.group.clone(), self.group_target.clone());
                        self.background(ctx, move |db| {
                            let count = db.move_group(&group, &split)?;
                            Ok(format!("Assigned {count} images to {split}"))
                        });
                    }
                });
            });
        if !open {
            self.selected_sample = None;
        }
    }

    fn sql_view(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Current SQL catalog");
        ui.monospace(self.db.path.display().to_string());
        ui.label(
            "Inspect every source, collection date, image resolution, file mapping, and split.",
        );
        ui.horizontal_wrapped(|ui| {
            for (title, sql) in [
                ("Images", DEFAULT_QUERY),
                (
                    "Sources",
                    "SELECT * FROM source_summary ORDER BY collected_at DESC;",
                ),
                ("Groups", "SELECT * FROM groups ORDER BY name;"),
                ("Labels", "SELECT * FROM labels ORDER BY id;"),
            ] {
                if ui.button(title).clicked() {
                    self.sql = sql.into();
                    match self.db.query(&self.sql) {
                        Ok(r) => self.result = Some(r),
                        Err(e) => self.report(Err(e)),
                    }
                }
            }
        });
        ui.add(
            egui::TextEdit::multiline(&mut self.sql)
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY)
                .desired_rows(6),
        );
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.job.is_none(), egui::Button::new("Run query"))
                .clicked()
            {
                match self.db.query(&self.sql) {
                    Ok(result) => {
                        self.result = Some(result);
                        self.status.clear();
                    }
                    Err(e) => self.report(Err(e)),
                }
            }
            if ui
                .add_enabled(self.job.is_none(), egui::Button::new("Apply updates"))
                .clicked()
            {
                let sql = self.sql.clone();
                self.background(ctx, move |db| {
                    db.apply_sql(&sql)?;
                    Ok("SQL updates validated and saved".into())
                });
            }
        });
        ui.label(RichText::new("Apply updates accepts UPDATE of source root_path/notes, sample image_path/mask_path, and group split. Paths must resolve to the same indexed files. To move files, use Sources → Manage location.").small().color(MUTED));
        if let Some(result) = &self.result {
            bento().show(ui, |ui| {
                egui::ScrollArea::both()
                    .max_height(350.0)
                    .id_salt("sql-results")
                    .show(ui, |ui| {
                        egui::Grid::new("sql-table").striped(true).show(ui, |ui| {
                            for column in &result.columns {
                                ui.label(RichText::new(column).strong());
                            }
                            ui.end_row();
                            for row in &result.rows {
                                for cell in row {
                                    ui.label(cell);
                                }
                                ui.end_row();
                            }
                        });
                    });
                ui.label(format!(
                    "{} rows{}",
                    result.rows.len(),
                    if result.truncated {
                        " · first 200 shown"
                    } else {
                        ""
                    }
                ));
            });
        }
        ui.collapsing("Live database schema", |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut self.schema)
                    .font(egui::TextStyle::Monospace)
                    .desired_width(f32::INFINITY)
                    .desired_rows(18)
                    .interactive(false),
            );
        });
        ui.collapsing("Save SQL document", |ui| {
            ui.label("Export the current schema and all metadata as a restorable SQL file.");
            ui.text_edit_singleline(&mut self.snapshot_path);
            if ui
                .add_enabled(self.job.is_none(), egui::Button::new("Save snapshot"))
                .clicked()
            {
                let path = PathBuf::from(&self.snapshot_path);
                self.background(ctx, move |db| {
                    db.save_snapshot(&path)?;
                    Ok(format!("Saved {}", path.display()))
                });
            }
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.draw(ctx);
    }
}

fn thumbnail(sample: &Sample, overlay: bool, labels: &[Label]) -> Result<egui::ColorImage> {
    let image_path = catalog::resolve(&sample.root, Path::new(&sample.image))?;
    let original = image::open(image_path)?;
    let mut preview = original.thumbnail(420, 280).to_rgb8();
    if overlay {
        let mask = catalog::read_mask(&catalog::resolve(&sample.root, Path::new(&sample.mask))?)?;
        anyhow::ensure!(
            (mask.width, mask.height) == (original.width(), original.height()),
            "Mask dimensions changed"
        );
        let palette: HashMap<_, _> = labels.iter().map(|l| (l.id as u8, l.color)).collect();
        let (width, height) = preview.dimensions();
        for (x, y, pixel) in preview.enumerate_pixels_mut() {
            let mx = x as u64 * mask.width as u64 / width as u64;
            let my = y as u64 * mask.height as u64 / height as u64;
            let id = mask.ids[(my * mask.width as u64 + mx) as usize];
            if id == 0 || id == 255 {
                continue;
            }
            let color = palette.get(&id).copied().unwrap_or([255, 0, 255]);
            for (channel, overlay) in pixel.0.iter_mut().zip(color) {
                *channel = ((*channel as u16 + overlay as u16) / 2) as u8;
            }
        }
    }
    Ok(egui::ColorImage::from_rgb(
        [preview.width() as usize, preview.height() as usize],
        preview.as_raw(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_native_view_produces_a_frame_without_data() {
        let temporary = tempfile::tempdir().unwrap();
        let db = Catalog::open(&temporary.path().join("catalog.sqlite3")).unwrap();
        let ctx = egui::Context::default();
        style(&ctx);
        let mut app = App::new(db, &ctx).unwrap();
        for tab in [Tab::Overview, Tab::Sources, Tab::Sql] {
            app.tab = tab;
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1320.0, 900.0),
                )),
                ..Default::default()
            };
            let output = ctx.run(input, |ctx| app.draw(ctx));
            assert!(!output.shapes.is_empty());
        }
    }

    #[test]
    fn source_and_image_inspectors_render_live_metadata() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("unrelated-folder-name");
        std::fs::create_dir(&root).unwrap();
        image::RgbImage::from_pixel(40, 20, image::Rgb([100, 100, 100]))
            .save(root.join("frame.png"))
            .unwrap();
        image::GrayImage::from_pixel(40, 20, image::Luma([1]))
            .save(root.join("frame.mask.png"))
            .unwrap();
        let mut db = Catalog::open(&temporary.path().join("data/catalog.sqlite3")).unwrap();
        let source = "hive-07__2026-06-12__camera-a";
        db.add_source(source, &root, "2026-06-12", "Collection notes")
            .unwrap();
        db.import(source, "val", "hive-07").unwrap();
        let ctx = egui::Context::default();
        style(&ctx);
        let mut app = App::new(db, &ctx).unwrap();
        app.selected_source = Some(source.into());
        app.selected_sample = Some(app.samples[0].clone());
        app.result = Some(app.db.query(DEFAULT_QUERY).unwrap());
        assert_eq!(app.result.as_ref().unwrap().rows[0][2], "2026-06-12");
        for tab in [Tab::Overview, Tab::Sources, Tab::Sql] {
            app.tab = tab;
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1320.0, 900.0),
                )),
                ..Default::default()
            };
            let output = ctx.run(input, |ctx| app.draw(ctx));
            assert!(!output.shapes.is_empty());
        }
    }

    #[test]
    fn mask_preview_is_bounded_and_uses_class_colors() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        image::RgbImage::from_pixel(1600, 800, image::Rgb([100, 100, 100]))
            .save(root.join("frame.png"))
            .unwrap();
        image::GrayImage::from_pixel(1600, 800, image::Luma([1]))
            .save(root.join("frame.mask.png"))
            .unwrap();
        let sample = Sample {
            id: 1,
            source: "source".into(),
            group: "hive".into(),
            split: "train".into(),
            root: root.to_owned(),
            image: "frame.png".into(),
            mask: "frame.mask.png".into(),
            collected: "2026-06-12".into(),
            width: 1600,
            height: 800,
        };
        let labels = vec![Label {
            id: 1,
            name: "worker".into(),
            color: [230, 168, 23],
            images: 1,
        }];
        let preview = thumbnail(&sample, true, &labels).unwrap();
        assert_eq!(preview.size, [420, 210]);
        assert_eq!(preview.pixels[0], Color32::from_rgb(165, 134, 61));
        let plain = thumbnail(&sample, false, &labels).unwrap();
        assert_eq!(plain.pixels[0], Color32::from_rgb(100, 100, 100));
    }
}
