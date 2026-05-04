"""Tests for dynamicauc/primitives/auctions.py: FirstPriceAuction, SecondPriceAuction."""
import pytest

from dynamicauc.primitives.auctions import FirstPriceAuction, SecondPriceAuction


class TestFirstPriceAuction:

    @pytest.fixture
    def auction(self):
        return FirstPriceAuction()

    def test_win_when_bid_exceeds_competing(self, auction):
        assert auction.win(bid=1.5, competing_bid=1.0) is True

    def test_lose_when_bid_below_competing(self, auction):
        assert auction.win(bid=0.5, competing_bid=1.0) is False

    def test_lose_on_tie(self, auction):
        assert auction.win(bid=1.0, competing_bid=1.0) is False

    def test_payment_equals_own_bid(self, auction):
        assert auction.payment(bid=1.5, competing_bid=1.0) == pytest.approx(1.5)

    def test_payment_independent_of_competing_bid(self, auction):
        assert auction.payment(bid=2.0, competing_bid=0.5) == pytest.approx(2.0)

    def test_zero_bid_loses(self, auction):
        assert auction.win(bid=0.0, competing_bid=0.1) is False

    def test_very_large_bid_wins(self, auction):
        assert auction.win(bid=1e9, competing_bid=100.0) is True

    @pytest.mark.parametrize("bid,comp", [(0.3, 0.1), (1.0, 0.9), (5.0, 0.01)])
    def test_payment_always_equals_bid(self, auction, bid, comp):
        assert auction.payment(bid=bid, competing_bid=comp) == pytest.approx(bid)


class TestSecondPriceAuction:

    @pytest.fixture
    def auction(self):
        return SecondPriceAuction()

    def test_win_when_bid_exceeds_competing(self, auction):
        assert auction.win(bid=1.5, competing_bid=1.0) is True

    def test_lose_when_bid_below_competing(self, auction):
        assert auction.win(bid=0.5, competing_bid=1.0) is False

    def test_lose_on_tie(self, auction):
        assert auction.win(bid=1.0, competing_bid=1.0) is False

    def test_payment_equals_competing_bid(self, auction):
        assert auction.payment(bid=1.5, competing_bid=1.0) == pytest.approx(1.0)

    def test_payment_always_equals_competing_bid(self, auction):
        assert auction.payment(bid=5.0, competing_bid=0.3) == pytest.approx(0.3)

    def test_payment_strictly_less_than_bid(self, auction):
        assert auction.payment(bid=2.0, competing_bid=0.5) < 2.0

    def test_zero_bid_loses(self, auction):
        assert auction.win(bid=0.0, competing_bid=0.1) is False

    @pytest.mark.parametrize("bid,comp", [(0.5, 0.1), (2.0, 0.9), (10.0, 3.0)])
    def test_payment_always_equals_competing(self, auction, bid, comp):
        assert auction.payment(bid=bid, competing_bid=comp) == pytest.approx(comp)
