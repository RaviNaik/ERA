<filename>sms.go
package objectia

import "github.com/objectia/objectia-go/internal"

// Send sends a SMS message
func (c *SMS) Send(from, to, text string) (*SMSReceipt, error) {
	var resp Response

	params := internal.NewParameters()
	params.Add("from", from)
	params.Add("to", to)
	params.Add("text", text)

	err := c.client.post("/v1/sms/send", params, &resp)
	if err != nil {
		return nil, err
	}

	result := &SMSReceipt{}
	err = internal.FromMap(resp.Data, result)
	if err != nil {
		return nil, err
	}

	return result, nil
}
