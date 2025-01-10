# Doc for nix-ld: https://github.com/nix-community/nix-ld
# https://github.com/mcdonc/.nixconfig/blob/master/videos/pydev/script.rst
# https://www.youtube.com/watch?v=7lVP4NJWJ9g
# https://github.com/Mic92/dotfiles/blob/main/machines/modules/fhs-compat.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { 
        inherit system; 
        overlays = [ rust-overlay.overlays.default ];
      };
      rustbin = pkgs.rust-bin.selectLatestNightlyWith (toolchain: toolchain.default.override {
        extensions = [ "rust-src" "rust-analyzer" "miri" ];
      });

      clangVersion = "19";
    in 
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          rustbin
          pkgs.cargo-show-asm
          pkgs.cargo-expand
          pkgs.cargo-flamegraph
          pkgs.cargo-valgrind
          pkgs.cargo-fuzz
          pkgs.cargo-pgo
          pkgs.rustfilt

          pkgs.openssl
          pkgs.pkg-config

          pkgs."clang_${clangVersion}"
          pkgs."llvmPackages_${clangVersion}".bintools
          pkgs."bolt_${clangVersion}"
          pkgs.cmake
        ];

        LIBCLANG_PATH = pkgs.lib.makeLibraryPath [ pkgs."llvmPackages_${clangVersion}".libclang.lib ];
      };
    };
}
