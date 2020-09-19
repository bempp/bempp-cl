from twitter import Twitter, OAuth
import sys

twitter = Twitter(
    auth = OAuth(*sys.argv[-4:]))

tweet = "Version " + sys.argv[-5] + " of Bempp-cl has just been released."
tweet += " https://github.com/bempp/bempp-cl/releases/tag/v" + sys.argv[-5]
twitter.statuses.update(status=tweet)
