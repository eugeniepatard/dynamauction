import pytest
from dynamicauc.primitives.auctions import FirstPriceAuction, SecondPriceAuction

def test_first_price_auction():
    auction = FirstPriceAuction()
    assert auction.win(bid=1.5, competing_bid=1.0) is True
    assert auction.win(bid=1.0, competing_bid=1.5) is False
    assert auction.payment(bid=1.5, competing_bid=1.0) == 1.5

def test_second_price_auction():
    auction = SecondPriceAuction()
    assert auction.win(bid=1.5, competing_bid=1.0) is True
    assert auction.win(bid=1.0, competing_bid=1.5) is False
    # Winner pays the highest competing bid
    assert auction.payment(bid=1.5, competing_bid=1.0) == 1.0
