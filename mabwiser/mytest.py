from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
arms = ['Arm1', 'Arm2', 'Arm3']
decisions = ['Arm1', 'Arm3', 'Arm2', 'Arm1']
rewards = [20, 17, 25, 9]
contexts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.1, 1.1]]
mab = MAB(arms, LearningPolicy.EpsilonGreedy(epsilon=0.1))
# mab = MAB(arms, LearningPolicy.UCB1())
# mab.fit(decisions, rewards, contexts=contexts)
mabs = [
    MAB(arms, LearningPolicy.Random()),
    MAB(arms, LearningPolicy.Popularity()),    
    MAB(arms, LearningPolicy.EpsilonGreedy(epsilon=0.1)),
    MAB(arms, LearningPolicy.UCB1()),
    MAB(arms, LearningPolicy.LinGreedy(epsilon=0.3)),
    MAB(arms, LearningPolicy.LinUCB(alpha=0.2)),
    MAB(arms, LearningPolicy.Softmax(tau=10.0)),
    # MAB(arms, LearningPolicy.ThompsonSampling(...)), # TODO, maybe, because it's binary
]
for mab in mabs:
    print(mab.learning_policy)
    mab.fit(decisions, rewards, contexts=contexts if mab.is_contextual else None)
    print(f"mab.predict_expectations(): {mab.predict_expectations(contexts)}")
    print(f"mab.predict_arm_proba(): {mab.predict_arm_proba(contexts)}")
    for _ in range(10):
        print(f"mab.predict(): {mab.predict(contexts)}")

    print("")
