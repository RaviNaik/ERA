<reponame>aholstenson/ataraxia-go
package ataraxia

import "github.com/fxamacker/cbor/v2"

// Message received from a Node in the network.
type Message struct {
	source      *Node
	messageType string
	data        cbor.RawMessage
}

// Source of this message. Can be used to reply.
func (m *Message) Source() *Node {
	return m.source
}

// Type of message received
func (m *Message) Type() string {
	return m.messageType
}

// Bind the data of this message into the given target
func (m *Message) Bind(target interface{}) error {
	return cbor.Unmarshal(m.data, target)
}
