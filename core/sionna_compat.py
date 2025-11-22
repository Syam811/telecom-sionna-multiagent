def phy_imports():
    """
    Compatibility layer for Sionna >=1.0 (new structure) and <1.0 (old).
    Returns: Constellation, Mapper, Demapper, AWGN, FlatFadingChannel, ebnodb2no
    """
    # New Sionna (1.x)
    try:
        from sionna.phy.mapping import Constellation, Mapper, Demapper
        from sionna.phy.channel import AWGN, FlatFadingChannel
        from sionna.phy.utils import ebnodb2no
        return Constellation, Mapper, Demapper, AWGN, FlatFadingChannel, ebnodb2no
    except Exception:
        pass

    # Old Sionna (0.x)
    from sionna.mapping import Constellation, Mapper, Demapper
    from sionna.channel import AWGN, FlatFadingChannel
    from sionna.utils import ebnodb2no
    return Constellation, Mapper, Demapper, AWGN, FlatFadingChannel, ebnodb2no
