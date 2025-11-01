{
  description = "GPT2 interpretability experiments";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      perSystem = {pkgs, system, ...}: {
        # Allow unfree and cuda-supported packages
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        devShells.default = pkgs.mkShellNoCC {
          packages = with pkgs; [
            uv
            python313

            # While uv handles most dependencies, Nix needs to manage ones with binaries due to its
            # non-FHS environment
            python313Packages.numpy
            python313Packages.torchWithCuda
          ];
        };
      };

      systems = inputs.nixpkgs.lib.systems.flakeExposed;
    };
}
