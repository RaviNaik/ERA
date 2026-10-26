package rabbitmq

import "github.com/streadway/amqp"

// Queue holds the parameters for a channel queue
type Queue struct {
	Name       string
	Channel    *amqp.Channel
	Queue      amqp.Queue
	Durable    bool
	AutoDelete bool
	Exclusive  bool
	NoWait     bool
	Args       amqp.Table
}

// Queue declares and returns a channel queue
func (ch *Channel) Queue(name string) (*Queue, error) {
	q := &Queue{
		Name:       name,
		Channel:    ch.Channel,
		Durable:    true,
		AutoDelete: false,
		Exclusive:  false,
		NoWait:     false,
		Args:       nil,
	}
	queue, err := ch.Channel.QueueDeclare(
		q.Name,
		q.Durable,
		q.AutoDelete,
		q.Exclusive,
		q.NoWait,
		q.Args,
	)
	q.Queue = queue

	return q, err
}

// Publish a message to queue
func (q *Queue) Publish(body []byte) error {
	err := q.Channel.Publish(
		message.Exchange,
		q.Queue.Name,
		message.Mandatory,
		message.Immediate,
		amqp.Publishing{
			DeliveryMode: message.DeliveryMode,
			ContentType:  message.ContentType,
			Body:         body,
		},
	)

	return err
}

// Consume messages from queue
func (q *Queue) Consume(config *ConsumerConfig) error {
	messages, err := q.Channel.Consume(
		q.Queue.Name,
		config.Consumer,
		config.AutoAck,
		config.Exclusive,
		config.NoLocal,
		config.NoWait,
		config.Args,
	)

	if err != nil {
		return err
	}

	for m := range messages {
		config.Callback(&Delivery{m})
	}

	return nil
}
