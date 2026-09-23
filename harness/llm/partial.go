package llm

import "context"

// PartialKind identifies a streamed fragment of an in-progress model response.
type PartialKind string

const (
	// PartialText is a delta of assistant message text.
	PartialText PartialKind = "text"
	// PartialReasoning is a delta of a reasoning summary.
	PartialReasoning PartialKind = "reasoning"
	// PartialReset discards fragments received so far; a new request attempt is starting.
	PartialReset PartialKind = "reset"
)

// Partial is a best-effort preview of a response while it streams. It is never persisted:
// the completed Response remains the only source of truth.
type Partial struct {
	Kind   PartialKind
	ItemID string
	Delta  string
	// Attempt identifies the request attempt that produced the partial. It increases with every attempt
	// in the process, so a consumer can discard partials from an attempt that has been superseded.
	Attempt uint64
}

// PartialSink receives partials. It is called synchronously from the adapter's stream loop
// and must not block.
type PartialSink func(Partial)

type partialSinkKey struct{}

// WithPartialSink returns a context whose model requests report partials to sink.
func WithPartialSink(ctx context.Context, sink PartialSink) context.Context {
	return context.WithValue(ctx, partialSinkKey{}, sink)
}

// PartialSinkFrom returns the sink installed by WithPartialSink, or nil.
func PartialSinkFrom(ctx context.Context) PartialSink {
	sink, _ := ctx.Value(partialSinkKey{}).(PartialSink)
	return sink
}
