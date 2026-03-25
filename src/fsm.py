"""
src/fsm.py — Context-aware Finite State Machine (FSM) for gesture filtering.

Implements the formal automaton A = <S, X, Y, δ, λ> from Section 3:

  S = {STOP, CRUISE, MANEUVER, HIGH_VIB}
  X = (v, a_long, a_lat, vib, gesture_id, confidence)
  Y = (y_exec, y_cmd)

The FSM prevents unintentional gesture activation during unsafe driving
states (sharp turns, high vibration) by filtering classifier outputs
through context-dependent allowlists and confidence thresholds.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

from config.settings import FSM_V_STOP, FSM_A_STAB, FSM_A_MAN, FSM_VIB_HIGH


class DrivingState(Enum):
    STOP      = 0   # S0 — vehicle stationary
    CRUISE    = 1   # S1 — stable forward motion
    MANEUVER  = 2   # S2 — cornering / sharp acceleration
    HIGH_VIB  = 3   # S3 — severe road vibration


class ContextFSM:
    """
    Context-aware gesture filter.

    State transitions δ(s, v, a_long, a_lat, vib) and output
    function λ(s, gesture_id, confidence) are defined by
    configurable thresholds and per-state gesture allowlists.

    Usage:
        fsm = ContextFSM()
        fsm.update_state(v=20.0, along=0.1, alat=0.1, vib=0.3)
        y_exec, y_cmd = fsm.decide(gesture_id=0, confidence=0.85)
    """

    # Default gesture allowlist per state
    # gesture IDs: 0=volume_up, 1=volume_down, 2=next_track, 3=prev_track
    DEFAULT_ALLOWED: Dict[DrivingState, List[int]] = {
        DrivingState.STOP:     [0, 1, 2, 3],
        DrivingState.CRUISE:   [0, 1, 2],
        DrivingState.MANEUVER: [],             # No gestures during manoeuvres
        DrivingState.HIGH_VIB: [0, 1],
    }

    # Minimum confidence threshold per state
    DEFAULT_CONFIDENCE: Dict[DrivingState, float] = {
        DrivingState.STOP:     0.60,
        DrivingState.CRUISE:   0.70,
        DrivingState.MANEUVER: 0.85,
        DrivingState.HIGH_VIB: 0.80,
    }

    def __init__(
        self,
        allowed: Optional[Dict[DrivingState, List[int]]] = None,
        confidence: Optional[Dict[DrivingState, float]] = None,
    ):
        self.state = DrivingState.STOP
        self.allowed    = allowed    or self.DEFAULT_ALLOWED
        self.confidence = confidence or self.DEFAULT_CONFIDENCE

    def update_state(
        self, v: float, along: float, alat: float, vib: float
    ) -> DrivingState:
        """
        Transition function δ — update driving state from sensor inputs.

        Args:
            v:     Vehicle speed (m/s).
            along: Longitudinal acceleration (m/s²).
            alat:  Lateral acceleration (m/s²).
            vib:   Vibration magnitude (normalised).

        Returns:
            Updated DrivingState.
        """
        if v < FSM_V_STOP and abs(along) < FSM_A_STAB and abs(alat) < FSM_A_STAB:
            self.state = DrivingState.STOP
        elif vib > FSM_VIB_HIGH:
            self.state = DrivingState.HIGH_VIB
        elif abs(alat) > FSM_A_MAN:
            self.state = DrivingState.MANEUVER
        else:
            self.state = DrivingState.CRUISE

        return self.state

    def decide(
        self, gesture_id: int, confidence: float
    ) -> Tuple[int, Optional[int]]:
        """
        Output function λ — decide whether to execute a recognised gesture.

        Args:
            gesture_id:  Integer class label from the classifier.
            confidence:  Softmax probability / confidence score (0–1).

        Returns:
            (y_exec, y_cmd):
                y_exec = 1 if gesture is executed, 0 if blocked.
                y_cmd  = gesture_id if executed, None if blocked.
        """
        allowed    = self.allowed.get(self.state, [])
        min_conf   = self.confidence.get(self.state, 1.0)

        if gesture_id in allowed and confidence >= min_conf:
            return 1, gesture_id
        return 0, None
