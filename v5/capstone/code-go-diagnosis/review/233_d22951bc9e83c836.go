<reponame>schafer14/purchase<filename>purchase_registries/inmemory/inmemory.go
package inmemory

import (
	"context"

	"github.com/schafer14/purchase"
)

type registry struct {
	orders   map[string]purchase.Order
	payments map[string]purchase.Payment
}

func New() purchase.PaymentRegistry {
	return &registry{
		orders:   map[string]purchase.Order{},
		payments: map[string]purchase.Payment{},
	}
}

func (r *registry) StoreOrder(ctx context.Context, order purchase.Order) error {
	r.orders[order.ID] = order

	return nil
}

func (r *registry) FindOrder(ctx context.Context, orderID purchase.OrderID) (purchase.Order, error) {
	return r.orders[orderID], nil
}

func (r *registry) OrdersByUser(ctx context.Context, userID purchase.UserID) ([]purchase.Order, error) {
	var results []purchase.Order
	for _, order := range r.orders {
		if order.UserID == userID {
			results = append(results, order)
		}
	}

	return results, nil
}

func (r *registry) StorePayment(ctx context.Context, payment purchase.Payment) error {
	r.payments[payment.ID] = payment

	return nil
}

func (r *registry) FindPayment(ctx context.Context, id purchase.PaymentID) (purchase.Payment, error) {
	return r.payments[id], nil
}

func (r *registry) PaymentsByUser(ctx context.Context, userID purchase.UserID) ([]purchase.Payment, error) {
	var results []purchase.Payment
	for _, payment := range r.payments {
		if payment.UserID == userID {
			results = append(results, payment)
		}
	}

	return results, nil
}
