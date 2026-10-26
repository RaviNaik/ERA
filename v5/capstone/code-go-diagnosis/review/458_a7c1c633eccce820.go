<filename>components/stream/handler/lb/lb.go
package lb

import (
	"context"

	"github.com/pipeproxy/pipe/components/balance"
	"github.com/pipeproxy/pipe/components/stream"
)

type LB struct {
	policy   balance.Policy
	handlers []stream.Handler
}

func NewLB(policy balance.Policy, handlers []stream.Handler) *LB {
	policy.Init(uint64(len(handlers)))
	return &LB{policy: policy, handlers: handlers}
}

func (l *LB) ServeStream(ctx context.Context, stm stream.Stream) {
	l.policy.InUse(func(i uint64) {
		l.handlers[i].ServeStream(ctx, stm)
	})
}
