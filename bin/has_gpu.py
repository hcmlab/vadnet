import re
from subprocess import CalledProcessError, check_output

if __name__ == '__main__':

	nvcc_version_regex = re.compile("release (?P<major>[0-9]+)\\.(?P<minor>[0-9]+)")
	use_gpu = False

	try:
		output = str(check_output(["nvcc", "--version"]))
		version_string = nvcc_version_regex.search(output)

		if version_string:
			major = int(version_string.group("major"))
			minor = int(version_string.group("minor"))

			# if major < 7:
				# print("detected incompatible CUDA version %d.%d" % (major, minor))
			# else:
				# print("detected compatible CUDA version %d.%d" % (major, minor))
			use_gpu = True

		#else:
		#	print("CUDA detected, but unable to parse version")
			
	#except CalledProcessError:
	#	print("no CUDA detected")
		
	except Exception as e:
		#print("error during CUDA detection: %s", e)
		pass

	print('-gpu' if use_gpu else '', end='')
