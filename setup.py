"""Installation script for the 'go2_rl_stairs_avoid' python package."""

import os
import toml

from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# unitree_rl_lab (isaaclab3.0 ブランチ) を別途 editable install しておくこと(README 参照)
INSTALL_REQUIRES = [
    "psutil",
]

setup(
    name="go2_rl_stairs_avoid",
    packages=["tools"],
    package_dir={"": "src"},
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache 2.0",
    include_package_data=True,
    python_requires=">=3.12",
    zip_safe=False,
)
