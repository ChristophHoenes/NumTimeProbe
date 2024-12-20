# parse arguments
POSITIONAL_ARGS=()

CREATE_SCREEN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -s|--screen)
      CREATE_SCREEN=true
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

# check argument for correct GPU selection
if [ $# -ne 1 ] ; then 
	echo "Please specify the GPU IDs to use as comma separated string e.g bash start_docker_lm_eval_run.sh 0,1"
fi
# start screen if specifed
if [ "$CREATE_SCREEN" == true ] ; then
  echo "starting screen"
	screen -S lm_eval_"$1"
fi

CONTAINER_NAME=lm_eval_$(date +%s)

echo "modify console script"
# save original script fife before editing it
cp scripts/console.sh scripts/.console_og.sh
# replace GPUs to select in console with the argument 
sed -i -E -e "s/DEVICES=.+\"$/DEVICES=\"$1\"/" scripts/console.sh
# change to non-interactive container (since we do not want to wait until the container closes and call docker exec for all commands)
sed -i -E -e 's/docker run -it \\$/docker run -itd \\/' scripts/console.sh
# no need for bash console
sed -i -E -e 's/^\s+bash$//' scripts/console.sh
# add container name
sed -i -E -e "s/\\\$IMAGE_TAG \\\\$/--name $CONTAINER_NAME \\\\\n\t\t\\\$IMAGE_TAG/" scripts/console.sh

echo "start container ($CONTAINER_NAME)"
# start the modified console container
bash scripts/console.sh
echo "Please check if container $CONTAINER_NAME is up before the next steps. Press any key to continue..." 
read -n 1 -s

echo "restore console script"
# restore original state of the console script
rm scripts/console.sh
mv scripts/.console_og.sh scripts/console.sh