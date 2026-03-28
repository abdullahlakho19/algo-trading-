"""
intelligence/volume_profile.py
─────────────────────────────────────────────────────────────────────────────
Volume Profile engine.
Computes Point of Control (POC), Value Area High (VAH),
Value Area Low (VAL), and tracks Naked POC levels.

These are institutional-grade levels where the most trading volume
occurred — price is magnetically attracted to them.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from core.logger import get_logger
from config import config

log = get_logger(__name__)


@dataclass
class VolumeProfileResult:
    """Complete volume profile for a given data window."""
    symbol:          str
    timeframe:       str
    poc:             float           # Point of Control — most traded price
    vah:             float           # Value Area High (70% of volume above POC)
    val:             float           # Value Area Low  (70% of volume below POC)
    value_area_pct:  float = 0.70
    profile:         pd.Series = field(default_factory=pd.Series)   # price → volume
    naked_pocs:      list[float] = field(default_factory=list)       # untested POCs
    is_bullish_va:   bool = False    # True if price is above POC
    is_bearish_va:   bool = False    # True if price is below POC
    current_price:   float = 0.0

    def __post_init__(self):
        if self.current_price > 0:
            self.is_bullish_va = self.current_price > self.poc
            self.is_bearish_va = self.current_price < self.poc

    @property
    def in_value_area(self) -> bool:
        return self.val <= self.current_price <= self.vah

    @property
    def above_value_area(self) -> bool:
        return self.current_price > self.vah

    @property
    def below_value_area(self) -> bool:
        return self.current_price < self.val


class VolumeProfileEngine:
    """
    Computes Volume Profile metrics from OHLCV data.

    Volume Profile shows where the most volume traded within a session
    or lookback window — these levels act as institutional reference points.
    """

    def __init__(self):
        self.num_bins          = config.volume_profile.NUM_BINS
        self.value_area_pct    = config.volume_profile.VALUE_AREA_PCT
        self.naked_poc_lookback = config.volume_profile.NAKED_POC_LOOKBACK
        self._naked_poc_history: dict[str, list[float]] = {}   # symbol → list of POCs

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        current_price: float | None = None,
    ) -> VolumeProfileResult:
        """
        Compute full volume profile for the given OHLCV DataFrame.

        Args:
            df:            OHLCV DataFrame
            symbol:        Instrument symbol
            timeframe:     e.g. "1h"
            current_price: Latest price (defaults to last close)

        Returns:
            VolumeProfileResult with POC, VAH, VAL, Naked POCs
        """
        if df.empty or len(df) < 20:
            log.warning(f"Insufficient data for volume profile: {symbol}")
            return self._empty_result(symbol, timeframe)

        price = current_price or float(df["close"].iloc[-1])

        # Build volume profile
        profile = self._build_profile(df)
        if profile.empty:
            return self._empty_result(symbol, timeframe)

        poc = self._find_poc(profile)
        vah, val = self._find_value_area(profile, poc)
        naked = self._find_naked_pocs(symbol, poc, price)

        result = VolumeProfileResult(
            symbol=symbol,
            timeframe=timeframe,
            poc=poc,
            vah=vah,
            val=val,
            value_area_pct=self.value_area_pct,
            profile=profile,
            naked_pocs=naked,
            current_price=price,
        )

        log.debug(
            f"{symbol} | VP | POC: {poc:.4f} VAH: {vah:.4f} "
            f"VAL: {val:.4f} | Naked POCs: {len(naked)}"
        )

        # Register this POC for future naked tracking
        self._register_poc(symbol, poc, price)

        return result

    # ── Profile Construction ──────────────────────────────────────────────────
    def _build_profile(self, df: pd.DataFrame) -> pd.Series:
        """
        Distribute volume across price bins using OHLCV candle ranges.
        Each candle's volume is split evenly across its high-low range.
        Returns a Series: price_level → volume
        """
        price_min = df["low"].min()
        price_max = df["high"].max()

        if price_min >= price_max:
            return pd.Series(dtype=float)

        bins  = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        volume_at_price = np.zeros(self.num_bins)

        for _, row in df.iterrows():
            lo, hi, vol = row["low"], row["high"], row["volume"]
            if hi <= lo or vol == 0:
                continue

            # Find which bins this candle covers
            lo_idx = np.searchsorted(bins, lo, side="left")
            hi_idx = np.searchsorted(bins, hi, side="right")
            lo_idx = max(0, lo_idx - 1)
            hi_idx = min(self.num_bins, hi_idx)

            num_bins_covered = hi_idx - lo_idx
            if num_bins_covered <= 0:
                continue

            vol_per_bin = vol / num_bins_covered
            volume_at_price[lo_idx:hi_idx] += vol_per_bin

        profile = pd.Series(volume_at_price, index=bin_centers)
        return profile

    # ── POC ───────────────────────────────────────────────────────────────────
    def _find_poc(self, profile: pd.Series) -> float:
        """Find the Point of Control — price with highest volume."""
        return float(profile.idxmax())

    # ── Value Area ────────────────────────────────────────────────────────────
    def _find_value_area(
        self, profile: pd.Series, poc: float
    ) -> tuple[float, float]:
        """
        Find Value Area High (VAH) and Value Area Low (VAL).
        The value area contains VALUE_AREA_PCT (default 70%) of total volume.
        Expands outward from POC.
        """
        total_volume  = profile.sum()
        target_volume = total_volume * self.value_area_pct

        poc_idx = profile.index.get_loc(poc) if poc in profile.index else profile.values.argmax()

        upper_idx = poc_idx
        lower_idx = poc_idx
        current_vol = profile.iloc[poc_idx]
        n = len(profile)

        while current_vol < target_volume:
            can_go_up   = upper_idx + 1 < n
            can_go_down = lower_idx - 1 >= 0

            if not can_go_up and not can_go_down:
                break

            vol_up   = profile.iloc[upper_idx + 1] if can_go_up   else -1
            vol_down = profile.iloc[lower_idx - 1] if can_go_down else -1

            if vol_up >= vol_down:
                upper_idx += 1
                current_vol += vol_up
            else:
                lower_idx -= 1
                current_vol += vol_down

        vah = float(profile.index[upper_idx])
        val = float(profile.index[lower_idx])
        return vah, val

    # ── Naked POC Tracking ────────────────────────────────────────────────────
    def _register_poc(self, symbol: str, poc: float, current_price: float) -> None:
        """Register a new POC for naked tracking."""
        if symbol not in self._naked_poc_history:
            self._naked_poc_history[symbol] = []
        self._naked_poc_history[symbol].append(poc)
        # Keep only recent history
        self._naked_poc_history[symbol] = (
            self._naked_poc_history[symbol][-self.naked_poc_lookback:]
        )

    def _find_naked_pocs(
        self, symbol: str, current_poc: float, current_price: float
    ) -> list[float]:
        """
        Return all historical POC levels that price has NOT revisited.
        These are 'naked' POCs — strong magnetic price targets.
        """
        history = self._naked_poc_history.get(symbol, [])
        if not history:
            return []

        # A POC is 'naked' if it hasn't been touched by current price
        # (simplified — in production track with actual price paths)
        naked = []
        for poc in history:
            if poc != current_poc:
                naked.append(poc)

        return sorted(naked)

    # ── Session Profile ───────────────────────────────────────────────────────
    def session_profile(
        self, df: pd.DataFrame, symbol: str, session: str = "today"
    ) -> VolumeProfileResult:
        """
        Compute volume profile for just the current/recent session.
        Useful for intraday analysis.
        """
        today = pd.Timestamp.utcnow().normalize()
        session_df = df[df.index >= today]

        if session_df.empty or len(session_df) < 5:
            # Fall back to last 50 candles
            session_df = df.tail(50)

        return self.analyse(session_df, symbol, "session")

    # ── Key Level Summary ─────────────────────────────────────────────────────
    def get_key_levels(self, result: VolumeProfileResult) -> list[dict]:
        """
        Return a structured list of all key VP levels.
        Used by the signal engine for confluence checks.
        """
        levels = [
            {"level": result.vah,  "type": "VAH",       "strength": "strong"},
            {"level": result.poc,  "type": "POC",        "strength": "strongest"},
            {"level": result.val,  "type": "VAL",        "strength": "strong"},
        ]
        for npoc in result.naked_pocs:
            levels.append({"level": npoc, "type": "NAKED_POC", "strength": "very_strong"})

        return sorted(levels, key=lambda x: x["level"])

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _empty_result(self, symbol: str, timeframe: str) -> VolumeProfileResult:
        return VolumeProfileResult(
            symbol=symbol, timeframe=timeframe,
            poc=0.0, vah=0.0, val=0.0,
        )


# Singleton
volume_profile_engine = VolumeProfileEngine()