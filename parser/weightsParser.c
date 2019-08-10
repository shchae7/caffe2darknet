void parse_weights(char *datacfg, char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
}

int main(int argc, char **argv)
{
    if(argc == 4)
    {
        parse_weights(argv[1], argv[2], argv[3]);
        return 0;
    }

    fprinf(stderr, "try: ./weightParser cfg/test.data cfg/test.cfg weights/test.weights\n")
    return 0;
}
