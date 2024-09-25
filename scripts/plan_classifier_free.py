import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import mlflow
import pytz
from datetime import datetime


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan_classifier_free')

experiment_name = args.dataset
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    # Create a new experiment if it doesn't exist
    mlflow.create_experiment(name=experiment_name)
mlflow.set_experiment(experiment_name)
mlflow.start_run(run_name=args.savepath.split("/", 2)[-1]) #remove logbase and dataset field
mlflow.log_params(args.as_dict())


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
# value_experiment = utils.load_diffusion(
#     args.loadbase, args.dataset, args.value_loadpath,
#     epoch=args.value_epoch, seed=args.seed,
# )

# ## ensure that the diffusion model and value function are compatible with each other
# utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
# value_function = value_experiment.ema
# guide_config = utils.Config(args.guide, model=value_function, verbose=False)
# guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    guidance_weight=args.guidance_weight,
    verbose=False,
)

logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env
observation = env.reset()

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(args.max_episode_length):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    state = env.state_vector().copy()

    ## format current observation for conditioning
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    ## print reward and score
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'values: {samples.values}',
        flush=True,
    )
    metrics = {
        'reward': reward,
        'total_reward': total_reward,
        'score': score,
    }
    mlflow.log_metrics(metrics, step=t)

    

    ## update rollout observations
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps
    logger.log(t, samples, state, rollout)

    if terminal:
        print("Terminated???")
        break

    observation = next_observation

## write results to json file at `args.savepath`
logger.finish(t, score, total_reward, terminal, diffusion_experiment, None)
mlflow.end_run()
