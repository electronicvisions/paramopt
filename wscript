import os
from os.path import join
from waflib.extras.test_base import summary
from waflib.extras.symwaf2ic import get_toplevel_path
from waflib import Utils


EXPERIMENT_NAME: str = "paramopt"

def depends(ctx):
    ctx("code-format")


def options(opt):
    opt.load("test_base")
    opt.load("pytest")


def configure(conf):
    conf.load("test_base")
    conf.load("pytest")

    conf.load("python")


def build(bld):
    bld.env.DLSvx_HARDWARE_AVAILABLE = "SLURM_HWDB_YAML" in os.environ

    bld(name=f"{EXPERIMENT_NAME}-pylib",
        features="py use pylint pycodestyle",
        source=bld.path.ant_glob("src/py/**/*.py"),
        relative_trick=True,
        install_path="${PREFIX}/lib",
        install_from="src/py",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        test_timeout=120)

    bld(name=f"{EXPERIMENT_NAME}-python_scripts",
        features="py use pylint pycodestyle",
        source=bld.path.ant_glob(f"src/py/{EXPERIMENT_NAME}/scripts/**/*.py"),
        relative_trick=True,
        install_path="${PREFIX}/bin",
        install_from=f"src/py/{EXPERIMENT_NAME}/scripts",
        chmod=Utils.O755,
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        use=[f"{EXPERIMENT_NAME}-pylib"],
        test_timeout=120)

    bld(name=f"{EXPERIMENT_NAME}-python_swtests",
        tests=bld.path.ant_glob("tests/sw/py/**/*.py"),
        features="use pytest pylint pycodestyle",
        use=[f"{EXPERIMENT_NAME}-pylib"],
        install_path="${PREFIX}/bin/tests/sw",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        test_timeout=160)

    bld.add_post_fun(summary)
