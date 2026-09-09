try:
    import gguf
    print('gguf version', gguf.__version__)
except Exception as e:
    print('no gguf', e)
