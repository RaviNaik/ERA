<reponame>klausbayrhammer/exercism.io-go<gh_stars>0
package twofer

func ShareWith(name string) string {
	return "One for " + geReceiverWithDefault(name) + ", one for me."
}

func geReceiverWithDefault(name string) string {
	if name == "" {
		return "you"
	}
	return name
}
