{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, fenix, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { 
        inherit system; 
        overlays = [ fenix.overlays.default ];
      };
      libPath = with pkgs; lib.makeLibraryPath [
        stdenv.cc.cc
      ];
    in 
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          (pkgs.fenix.complete.withComponents [
              "cargo"
              "clippy"
              "rust-src"
              "rustc"
              "rustfmt"
              "miri"
          ])
          pkgs.cargo-show-asm
          pkgs.cargo-expand
          pkgs.cargo-flamegraph
          pkgs.cargo-valgrind
          pkgs.cargo-fuzz
          pkgs.cargo-pgo

          pkgs.rust-analyzer-nightly
          pkgs.openssl
          pkgs.pkg-config

          pkgs.clang_19
          pkgs.llvmPackages_19.bintools
          pkgs.bolt_19
          pkgs.rustfilt
        ];

        #LD_LIBRARY_PATH = libPath;
        shellHook = ''
          export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
        '';

      };
    };
}
