"""
Hydra search path plugin for bids2table

See the `hydra plugins example`_

.. _hydra plugins example: https://github.com/facebookresearch/hydra/tree/main/examples/plugins/example_searchpath_plugin
"""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class BIDS2TableSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        # Note that bids2table.config is outside of the plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        search_path.append(
            provider="bids2table-searchpath-plugin", path="pkg://bids2table.config"
        )
