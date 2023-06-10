run_main() {
    python run.py \
    --content_img_path "$1" \
    --style_img_path "$2" \
    --content_weight "$3" \
    --style_weight "$4"
}

############# Hyperparameter Tuning ##############
# run_main ./images/content/tubingen.jpeg ./images/style/frida_kahlo.jpeg 1 1e4
# run_main ./images/content/tubingen.jpeg ./images/style/frida_kahlo.jpeg 1 1e5
# run_main ./images/content/tubingen.jpeg ./images/style/frida_kahlo.jpeg 1 1e6

############# Exchange Content and Style ##############
# run_main ./images/style/frida_kahlo.jpeg ./images/content/tubingen.jpeg 1 1e5

############# Style Transfer ##############
# run_main ./images/content/phipps.jpeg ./images/style/frida_kahlo.jpeg 1 1e5
# run_main ./images/content/tubingen.jpeg ./images/style/frida_kahlo.jpeg 1 1e5
# run_main ./images/content/fallingwater.png ./images/style/frida_kahlo.jpeg 1 1e5

# run_main ./images/content/phipps.jpeg ./images/style/starry_night.jpeg 1 1e5
# run_main ./images/content/tubingen.jpeg ./images/style/starry_night.jpeg 1 1e5
# run_main ./images/content/fallingwater.png ./images/style/starry_night.jpeg 1 1e5

# run_main ./images/content/phipps.jpeg ./images/style/picasso.jpg 1 1e5
# run_main ./images/content/tubingen.jpeg ./images/style/picasso.jpg 1 1e5
# run_main ./images/content/fallingwater.png ./images/style/picasso.jpg 1 1e5
