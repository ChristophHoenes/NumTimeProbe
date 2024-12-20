# assign variables if not yet set in environment
: "${UID:=$(id -u)}"
: "${GID:=$(id -g)}"
: "${USERNAME:=$(whoami)}"
: "${IS_ROOTLESS_DOCKER:=false}"

docker build -t choenes/num_tab_qa --build-arg UID=$UID --build-arg GID=$GID --build-arg USERNAME=$USERNAME --build-arg IS_ROOTLESS_DOCKER=$IS_ROOTLESS_DOCKER .