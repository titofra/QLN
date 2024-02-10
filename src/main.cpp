#include "setup.hpp"

int main (void) {
    srand ((int) time (0));	// init seed for rand function

    // constants
    const unsigned int N_CALLS_LISTEN = 44100 * 2;
    const unsigned int N_CALLS_GUESS = 18000;
    const unsigned int MAX_THREADS = 0; // default
    const std::string rootPath = "dataset/data/wav/";

    // init pneatm logger
	spdlog::set_pattern ("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    auto logger = spdlog::stdout_color_mt("console");
    //auto logger = spdlog::rotating_logger_mt("AGPNet_logger", "logs/log.txt", 1048576 * 100, 500);

    // init populations
    unsigned int popSize_gen = 50;
    unsigned int popSize_dis = 50;
    pneatm::Population<double> generators = SetupPopulation (popSize_gen, logger.get (), "save/stats_gen.csv");
    pneatm::Population<double> discriminators = SetupPopulation (popSize_dis, logger.get (), "save/stats_dis.csv");
    //pneatm::Population<double> pop = LoadPopulation ("save/8499", logger.get (), "");

    // init mutation parameters
    std::function<pneatm::mutationParams_t (double)> paramsMap_gen = SetupMutationParametersMaps ();
    std::function<pneatm::mutationParams_t (double)> paramsMap_dis = SetupMutationParametersMaps ();

    bool generator_is_winning = false;
    std::unique_ptr<pneatm::Genome<double>> bestGenerator = generators.getGenome (0).clone ();
    std::unique_ptr<pneatm::Genome<double>> bestDiscriminator = discriminators.getGenome (0).clone ();

    while (generators.getGeneration () + discriminators.getGeneration () < 9500) { // while goal is not reach
        logger->info ("generation {}", generators.getGeneration () + discriminators.getGeneration ());

        // Prepare the samples
        std::vector<double> sample = getRandomSample (rootPath);
        std::vector<double> sample_listen (sample.begin (), sample.begin () + N_CALLS_LISTEN);
        std::vector<double> sample_target (sample.begin (), sample.begin () + N_CALLS_LISTEN + N_CALLS_GUESS);

        if (generator_is_winning) {


            /* TEST the discriminators on a GENERATED AUDIO */

            // Run the networks over the listening part e.g. it don't predict anything, it just listen for the audio.
            bool locked = false;
            for (const double& input : sample_listen) {
				if (!locked) {
                    bestGenerator->loadInput (input, 0);
                    locked = bestGenerator->runNetwork ();
                }
			}

            // Run the network over the predicting part e.g. it predict the audio.
            std::vector<double> generated_audio;
            for (unsigned int k = 0; k < N_CALLS_GUESS; k++) {
				if (!locked) {
                    generated_audio.push_back (bestGenerator->template getOutput<double> (0));
                    bestGenerator->loadInput (generated_audio.back (), 0);
                    locked = bestGenerator->runNetwork ();
                }
			}

            // reset genome's memory
            bestGenerator->resetMemory ();

            std::vector<std::vector<void*>> dis_inputs;
            std::vector<double> dis_guess_on_generated;
            if (!locked) {
                // get the resulting audio
                std::vector<double> audio = sample_listen;
                audio.insert (audio.end (), generated_audio.begin (), generated_audio.end ());

                // setup discriminator's inputs
                for (double& input : audio) {
                    dis_inputs.push_back (std::vector<void*> (1, static_cast<void*> (&input)));
                }

                // run the discriminators
                discriminators.run (dis_inputs, nullptr, MAX_THREADS);

                // get discriminators's output
                for (unsigned int id = 0; id < popSize_dis; id++) {
                    // get the probability of the generated audio. Note that an ideal discriminator's output is 0 as it is not a real audio.
                    dis_guess_on_generated.push_back (discriminators.getGenome (id).template getOutput<double> (0));
                }

                // reset the discriminators's memory
                discriminators.resetMemory ();

            } else {
                // the best generator raised a NaN, we will give to each discriminators the maximum fitness
                for (unsigned int id = 0; id < popSize_dis; id++) {
                    // get the probability of the generated audio. Note that an ideal discriminator's output is 0 as it is not a real audio.
                    dis_guess_on_generated.push_back (0.0);
                }
            }


            /* TEST the discriminators on a REAL AUDIO */

            // setup discriminator's inputs
            dis_inputs.clear ();
            for (double& input : sample_target) {
                dis_inputs.push_back (std::vector<void*> (1, static_cast<void*> (&input)));
            }

            // run the discriminators
            discriminators.run (dis_inputs, nullptr, MAX_THREADS);

            // get discriminators's output
            std::vector<double> dis_guess_on_real;
            for (unsigned int id = 0; id < popSize_dis; id++) {
                // get the probability of the real audio. Note that an ideal discriminator's output is 1 as it is a real audio.
                dis_guess_on_real.push_back (discriminators.getGenome (id).template getOutput<double> (0));
            }

            // reset the discriminators's memory
            discriminators.resetMemory ();


            /* GRADE the discriminators */
            for (unsigned int id = 0; id < popSize_dis; id++) {
                const double score =
                    (1.0 - dis_guess_on_real [id])
                    + dis_guess_on_generated [id];
                // grade the discriminator
                std::cout << score << "   ";
                if (score > 0.0) {
                    std::cout << 1.0 / score << std::endl;
                    discriminators.getGenome (id).setFitness (1.0 / score);
                } else {
                    std::cout << DBL_MAX << std::endl;
                    discriminators.getGenome (id).setFitness (DBL_MAX);
                }
            }


            /* SPECIATE the discriminators */
            discriminators.speciate (5, 100, 0.3);


            /* SAVE */
            bestDiscriminator = discriminators.getGenome (-1).clone ();
            discriminators.save ("save/" + std::to_string (generators.getGeneration () + discriminators.getGeneration ()) + "_dis");


            /* generate the NEW GENERATION of discriminators */
            discriminators.crossover (true, 0.7);


            /* MUTATE the discriminators */
            discriminators.mutate (paramsMap_dis);


            /* CHECK if the generator is still better than the discriminators */
            if (dis_guess_on_real [bestDiscriminator->getID ()] > 0.5f && dis_guess_on_generated [bestDiscriminator->getID ()] < 0.5f) {
                generator_is_winning = false;
            }

            /* LOG */
            logger->info ("dis_loss_real {} \t dis_loss_gen {}", (1.0f - dis_guess_on_real [bestDiscriminator->getID ()]), dis_guess_on_generated [bestDiscriminator->getID ()]);


        } else {


            /* generators GENERATE audio */
        
            std::vector<std::vector<void*>> sample_listen_inputs;
            for (double& value : sample_listen) {
                sample_listen_inputs.push_back (std::vector<void*> {static_cast<void*> (&value)});
            }

            // Run the networks over the listening part e.g. it don't predict anything, it just listen for the audio.
            generators.run (sample_listen_inputs, nullptr, MAX_THREADS);

            // Run the networks over the predicting part e.g. it predict the audio.
            std::vector<std::vector<void*>> generated_audio;
            generators.run (N_CALLS_GUESS, &generated_audio, MAX_THREADS);

            // reset genomes's memory
            generators.resetMemory ();


            /* the discriminator GRADE the generated audio */

            for (std::pair<const unsigned int, std::unique_ptr<Genome<double>>>& generator : generators) {

                // setup discriminator's inputs
                std::vector<std::vector<void*>> dis_inputs = sample_listen_inputs;
                dis_inputs.insert (dis_inputs.end (), generated_audio.begin (), generated_audio.end ());

                // run the discriminator
                bool locked = false;
                for (const double& input : sample_listen) {
                    if (!locked) {
                        bestDiscriminator->loadInput (input, 0);
                        locked = bestDiscriminator->runNetwork ();
                    }
                }
                for (const void* input : generated_audio [generator.second->getID ()]) {
                    if (!locked) {
                        bestDiscriminator->loadInput (input, 0);
                        locked = bestDiscriminator->runNetwork ();
                    }
                }

                if (!locked) {
                    const double score = (1.0 - bestDiscriminator->template getOutput<double> (0));
                    // grade the generator
                    std::cout << score << "   ";
                    if (score > 0.0) {
                        std::cout << 1.0 / score << std::endl;
                        generator.second->setFitness (1.0 / score);
                    } else {
                        std::cout << DBL_MAX << std::endl;
                        generator.second->setFitness (DBL_MAX);
                    }
                }

                // reset the discriminator's memory
                bestDiscriminator->resetMemory ();

            }


            /* SPECIATE the generators */
            generators.speciate (5, 100, 0.3);


            /* SAVE */
            bestGenerator = generators.getGenome (-1).clone ();
            generators.save ("save/" + std::to_string (generators.getGeneration () + discriminators.getGeneration ()) + "_gen");


            /* generate the NEW GENERATION of generators */
            generators.crossover (true, 0.7);


            /* MUTATE the generators */
            generators.mutate (paramsMap_gen);


            /* CHECK if the discriminator is still better than the generators */
            if (bestGenerator->getFitness () > 2.0) {
                generator_is_winning = true;
            }

            /* LOG */
            logger->info ("gen_loss {}", 1.0 / bestGenerator->getFitness ());


        }
    }
    return 0;
}