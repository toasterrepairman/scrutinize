{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    rust-overlay,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (system: let
      overlays = [(import rust-overlay)];
      pkgs = import nixpkgs {inherit system overlays;};
      rustVersion = pkgs.rust-bin.stable.latest.default;
      packageDeps = with pkgs; [
        openssl.dev
        rustc
        cargo
        cairo
        gdk-pixbuf
        gobject-introspection
        graphene
        gtk3.dev
        gtksourceview5
        libadwaita.dev
        hicolor-icon-theme
        pandoc
        pango
        pkg-config
        appstream-glib
        polkit
        gettext
        desktop-file-utils
        meson
        git
        wrapGAppsHook4
        libsecret
      ];

      rustPlatform = pkgs.makeRustPlatform {
        cargo = rustVersion;
        rustc = rustVersion;
      };

      myRustBuild = rustPlatform.buildRustPackage {
        pname = "scrutinize"; # make this what ever your cargo.toml package.name is
        version = "0.1.0";
        src = ./.; # the folder with the cargo.toml
        nativeBuildInputs = packageDeps;
        buildInputs = packageDeps;

        cargoLock.lockFile = ./Cargo.lock;

        postInstall = ''
            # Install desktop file
            install -Dm644 resources/com.github.scrutinize.desktop $out/share/applications/com.github.scrutinize.desktop

            # Install MIME type definition
            install -Dm644 resources/scrutinize-mimetypes.xml $out/share/mime/packages/scrutinize.xml

            install -Dt $out/share/icons resources/icon-scrutinize.png

            # Update MIME and desktop databases
            ${pkgs.shared-mime-info}/bin/update-mime-database $out/share/mime
            ${pkgs.desktop-file-utils}/bin/update-desktop-database $out/share/applications
        '';
      };

    in {
      defaultPackage = myRustBuild;
      devShell = pkgs.mkShell {
        nativeBuildInputs = packageDeps;
        # CUDA_ROOT = "${pkgs.cudaPackages.cudatoolkit}";
        buildInputs = [(rustVersion.override {extensions = ["rust-src"];})];
      };

      meta = with nixpkgs.lib; {
        description = "scrutinize";
        license = licenses.gpl3;
        platforms = platforms.all;
      };
    });
}
