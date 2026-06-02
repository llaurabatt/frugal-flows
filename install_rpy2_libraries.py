import os
import platform
import shutil
import sys

import rpy2.robjects as ro
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr

# Toggle this to True if you want detailed output
VERBOSE = False


def _r_fortran_compiler():
    """Resolve the Fortran compiler R was actually built with.

    Queries `R CMD config FC` through the already-loaded R session rather than
    assuming a hardcoded toolchain path. This is correct for both system R
    (macOS: /opt/gfortran/bin/gfortran) and conda R (its bundled
    arm64-apple-darwin*-gfortran on $CONDA_PREFIX/bin). Returns the resolved
    absolute path to the compiler, or None if R reports none / it can't be
    found on disk.
    """
    try:
        fc_vec = ro.r('''
            r_bin <- file.path(R.home("bin"), "R")
            out <- tryCatch(
                system2(r_bin, c("CMD", "config", "FC"), stdout = TRUE, stderr = FALSE),
                error = function(e) character(0)
            )
            paste(out, collapse = " ")
        ''')
        fc = str(fc_vec[0]).strip() if len(fc_vec) else ""
    except Exception:
        fc = ""

    if not fc:
        return None

    # FC may be a bare program name (resolved via PATH — conda puts
    # $CONDA_PREFIX/bin on PATH) or an absolute path, optionally followed by
    # flags. Take the first token and resolve it.
    compiler = fc.split()[0]
    if os.path.isabs(compiler):
        return compiler if os.path.exists(compiler) else None
    return shutil.which(compiler)


def check_system_prerequisites():
    """Verify system-level tools that causl's compile pipeline depends on.

    The Fortran-compiler check trusts R's own build configuration
    (`R CMD config FC`) instead of a hardcoded path, so it passes on both
    system R and conda R. See .claude/upstream_causl_patch.md for rationale.
    """
    system = platform.system()
    missing = []

    # gfortran: causl ships Fortran source (mvt.f). Ask R which compiler it
    # will actually use rather than guessing where it lives on disk.
    fc = _r_fortran_compiler()
    if fc is None:
        # Fall back to the legacy hardcoded-path / PATH check only when R
        # can't report a usable compiler itself.
        if system == "Darwin":
            if not os.path.exists("/opt/gfortran/bin/gfortran"):
                missing.append(
                    "No Fortran compiler found: R's `R CMD config FC` returned nothing usable\n"
                    "    and /opt/gfortran/bin/gfortran (where system R's Makeconf expects it) is absent.\n"
                    "    Fix: download gfortran-12.2-universal.pkg (Intel + Apple Silicon, R 4.3.0–4.4.3) from\n"
                    "         https://github.com/R-macos/gcc-12-branch/releases/tag/12.2-darwin-r0.1\n"
                    "         (also linked from https://mac.r-project.org/tools/)\n"
                    "         then install with: sudo installer -pkg ~/Downloads/gfortran-12.2-universal.pkg -target /"
                )
        elif system == "Linux":
            if shutil.which("gfortran") is None:
                missing.append(
                    "No Fortran compiler found (R reported none and gfortran is not on PATH).\n"
                    "    Fix: sudo apt-get install gfortran  (or your distro's equivalent)"
                )
    elif VERBOSE:
        print(f"R Fortran compiler: {fc}")

    # System GSL: required by the R 'gsl' package, which is a hard dep of causl.
    # On conda, gsl-config lives in $CONDA_PREFIX/bin (on PATH when the env is
    # active); look there too before declaring it missing.
    gsl_config = shutil.which("gsl-config")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if gsl_config is None and conda_prefix:
        candidate = os.path.join(conda_prefix, "bin", "gsl-config")
        if os.path.exists(candidate):
            gsl_config = candidate
    if gsl_config is None:
        if system == "Darwin":
            missing.append(
                "System GSL (GNU Scientific Library) not found.\n"
                "    Fix: brew install gsl  (or `micromamba install -n <env> gsl` for conda R)"
            )
        elif system == "Linux":
            missing.append(
                "System GSL (GNU Scientific Library) not found.\n"
                "    Fix: sudo apt-get install libgsl-dev  (or `micromamba install -n <env> gsl`)"
            )
        else:
            missing.append("System GSL (GNU Scientific Library) not found on PATH.")

    if missing:
        print("=" * 72, file=sys.stderr)
        print("MISSING SYSTEM PREREQUISITES", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        for m in missing:
            print(f"- {m}\n", file=sys.stderr)
        print("Install the items above, then re-run this script.", file=sys.stderr)
        sys.exit(1)

    print(f"System prerequisites OK: Fortran compiler ({fc or 'gfortran'}) + GSL detected.")

# Utility functions for suppressing and enabling R output
def suppress_r_output():
    """
    Suppress R messages, warnings, and output.
    """
    ro.r('sink("/dev/null")')  # Redirect all R output to /dev/null
    ro.r('suppressMessages(suppressWarnings(sink()))')

def enable_r_output():
    """
    Re-enable R output.
    """
    ro.r('sink()')  # Re-enable R output

# Function to set CRAN mirror automatically
def set_cran_mirror():
    """
    Automatically set CRAN mirror to the first in the list.
    """
    utils = importr('utils')
    try:
        if not VERBOSE:
            suppress_r_output()
        print("Setting CRAN mirror to the first available option...")
        utils.chooseCRANmirror(ind=1)
        print("CRAN mirror set successfully.")
    except Exception as e:
        print(f"Error setting CRAN mirror: {e}")
    finally:
        if not VERBOSE:
            enable_r_output()

# Function to ensure 'remotes' is installed
def ensure_remotes_installed():
    """
    Ensure the 'remotes' package is installed.
    """
    utils = importr('utils')
    try:
        if not VERBOSE:
            suppress_r_output()
        print("Installing 'remotes' package if not already installed...")
        utils.install_packages(StrVector(["remotes"]), repos="https://cloud.r-project.org")
        print("'remotes' package is ready.")
    except Exception as e:
        print(f"Error installing 'remotes': {e}")
    finally:
        if not VERBOSE:
            enable_r_output()

# Function to install a specific version of an R package
def install_specific_version(package_name, version):
    """
    Install a specific version of an R package using remotes::install_version.
    :param package_name: Name of the R package
    :param version: Version of the package to install
    """
    try:
        if not VERBOSE:
            suppress_r_output()
        print(f"Installing {package_name} version {version}...")
        ro.r(f'remotes::install_version("{package_name}", version = "{version}", repos = "https://cloud.r-project.org")')
        print(f"{package_name} version {version} installed successfully.")
    except Exception as e:
        print(f"Error installing {package_name} version {version}: {e}")
    finally:
        if not VERBOSE:
            enable_r_output()

# Function to install causl from GitHub
def install_causl():
    """
    Install the causl package from GitHub.
    """
    try:
        if not VERBOSE:
            suppress_r_output()
        print("Installing causl package from GitHub...")
        ro.r('remotes::install_github("rje42/causl")')
        # remotes::install_github does not raise on R-side compile/lazy-load failure;
        # the only honest way to know it worked is to actually load it.
        ro.r('library(causl)')
        print("causl package installed successfully.")
    except Exception as e:
        print(f"Error installing causl: {e}")
    finally:
        if not VERBOSE:
            enable_r_output()

# Function to verify if a package is installed
def is_package_installed(package_name):
    """
    Check if an R package is installed and load it.
    :param package_name: Name of the R package
    """
    try:
        if not VERBOSE:
            suppress_r_output()
        ro.r(f"library({package_name})")
        print(f"Package '{package_name}' is installed and loaded successfully.")
        return True
    except Exception as e:
        print(f"Package '{package_name}' is not installed or failed to load: {e}")
        return False
    finally:
        if not VERBOSE:
            enable_r_output()

# Function to test causl methods
def test_causl_methods():
    """
    Test if causl methods are working correctly.
    """
    try:
        print("Testing causl methods...")
        if not VERBOSE:
            suppress_r_output()
        ro.r("""
                library(causl)
                pars <- list(Zc1 = list(beta = c(1), phi=1),
                             Zc2 = list(beta = c(1), phi=1),
                             Zc3 = list(beta = c(1), phi=1),
                             Zc4 = list(beta = c(1), phi=1),
                             X = list(beta = c(-2,1,1,1,1)),
                             Y = list(beta = c(1, 1), phi=1),
                             cop = list(beta=matrix(c(0.5,0.3,0.1,0.8,
                                                          0.4,0.1,0.8,
                                                              0.1,0.8,
                                                                  0.8), nrow=1)))
                
                set.seed(1)  # for consistency
                fams <- list(c(3,3,3,3),5,1,1)
                data_samples <- causalSamp(10, formulas=list(list(Zc1~1, Zc2~1, Zc3~1, Zc4~1), X~Zc1+Zc2+Zc3+Zc4, Y~X, ~1), family=fams, pars=pars)
        """)
        print("causl methods are working correctly.")
    except Exception as e:
        print(f"Error testing causl methods: {e}")
        exit(1)
    finally:
        if not VERBOSE:
            enable_r_output()

# Main installation script
def main():
    # Step 0: System prerequisites (gfortran, system GSL).
    # causl ships Fortran source (mvt.f) and depends on the R 'gsl' package,
    # which in turn needs the system GSL library. Both are external to R.
    # The check is now conda-aware (trusts `R CMD config FC` + looks in
    # $CONDA_PREFIX for gsl-config), so it passes on both system and conda R.
    check_system_prerequisites()

    # Step 1: Set CRAN mirror
    set_cran_mirror()

    # Step 2: Ensure 'remotes' is installed
    ensure_remotes_installed()

    # Step 3: Install specific versions of MASS and Matrix
    install_specific_version("MASS", "7.3-60")
    install_specific_version("Matrix", "1.6-5")

    # Step 4: Install R 'gsl' explicitly. CRAN's current gsl (2.1-9) requires
    # R >= 4.5.0; 2.1-8 is the most recent version compatible with R 4.4.x
    # and still loads cleanly on newer R. Installing it before causl prevents
    # causl's dep resolution from silently skipping gsl as "not available".
    install_specific_version("gsl", "2.1-8")

    # Step 5: Install causl from GitHub
    install_causl()

    # Step 6: Verify all packages
    if (
        is_package_installed("MASS")
        and is_package_installed("Matrix")
        and is_package_installed("gsl")
        and is_package_installed("causl")
    ):
        print("All packages installed and verified successfully.")
    else:
        print("Some packages failed to install or verify.")
        exit(1)

    # Step 7: Test causl methods
    test_causl_methods()

# Run the script
if __name__ == "__main__":
    main()
