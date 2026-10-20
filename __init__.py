def comfy_entrypoint():
    from .nodes import MOSSImageBlockCacheT8Extension
    return MOSSImageBlockCacheT8Extension()

