<filename>crypto/cipher/noop/docs.go<gh_stars>100-1000
// Package noop is a no operation encryption algorithm
// for use when the message is intended to be publically readable.
//
// Noop implements the Encrypt and Decrypt interfaces to ensure that is is compatible
// with Mailchain protocol.
// It does not perform any encryption or decryption actions.
package noop
