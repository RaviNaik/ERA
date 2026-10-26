<filename>logz/tracing_test.go<gh_stars>0
package logz

import (
	"strings"
	"testing"

	"github.com/getsentry/sentry-go"
	"github.com/stretchr/testify/require"
)

func TestParseSentryTraceHeader(t *testing.T) {
	traceID, parentSpanID, sampled, ok := parseSentryTraceHeader("0123456789abcdef0123456789abcdef-0123456789abcdef-1")
	require.True(t, ok)
	require.Equal(t, "0123456789abcdef0123456789abcdef", traceID.String())
	require.Equal(t, "0123456789abcdef", parentSpanID.String())
	require.True(t, sampled.Bool())

	traceID, parentSpanID, sampled, ok = parseSentryTraceHeader("0123456789abcdef0123456789abcdef-0123456789abcdef-0")
	require.True(t, ok)
	require.Equal(t, "0123456789abcdef0123456789abcdef", traceID.String())
	require.Equal(t, "0123456789abcdef", parentSpanID.String())
	require.False(t, sampled.Bool())

	traceID, parentSpanID, sampled, ok = parseSentryTraceHeader("bad")
	require.False(t, ok)
	require.Equal(t, sentry.TraceID{}, traceID)
	require.Equal(t, sentry.SpanID{}, parentSpanID)
	require.False(t, sampled.Bool())
}

func TestTraceBeforeSend(t *testing.T) {
	require.Nil(t, traceBeforeSend(nil))
	require.Equal(t, sentry.NewEvent(), traceBeforeSend(sentry.NewEvent()))

	event := sentry.NewEvent()
	req := &sentry.Request{Method: "GET"}
	event.Extra[logsRequestExtraKey] = req
	traceBeforeSend(event)
	require.Empty(t, event.Extra)
	require.Equal(t, req, event.Request)

	event = sentry.NewEvent()
	req = &sentry.Request{Method: "GET", Data: strings.Repeat("x", sentryMaxRequestBodyBytes+1)}
	event.Extra[logsRequestExtraKey] = req
	traceBeforeSend(event)
	require.Empty(t, event.Extra)
	require.Equal(t, req, event.Request)
	require.Empty(t, event.Request.Data)
}
