def test_package_exposes_version():
    import fanndyn

    assert isinstance(fanndyn.__version__, str)
    assert fanndyn.__version__
