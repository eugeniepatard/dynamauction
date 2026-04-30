"""
Auction rules determining payment from bid and competing bid.

``FirstPriceAuction``: winner pays their own bid.
``SecondPriceAuction``: winner pays the highest competing bid (Vickrey).
"""
from abc import ABC, abstractmethod


class AuctionRule(ABC):
    @abstractmethod
    def win(self, bid: float, competing_bid: float) -> bool: ...

    @abstractmethod
    def payment(self, bid: float, competing_bid: float) -> float: ...


class FirstPriceAuction(AuctionRule):
    """Winner pays their own bid."""
    def win(self, bid: float, competing_bid: float) -> bool:
        return bid > competing_bid

    def payment(self, bid: float, competing_bid: float) -> float:
        return bid


class SecondPriceAuction(AuctionRule):
    """Winner pays the highest competing bid."""
    def win(self, bid: float, competing_bid: float) -> bool:
        return bid > competing_bid

    def payment(self, bid: float, competing_bid: float) -> float:
        return competing_bid
