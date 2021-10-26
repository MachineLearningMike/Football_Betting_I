

def odds_loss(y_true, y_pred):
    """
    The function implements the custom loss function
    
    Inputs
    true : a vector of dimension batch_size, 7. A label encoded version of the output and the backp1_a and backp1_b
    pred : a vector of probabilities of dimension batch_size , 5.
    
    Returns 
    the loss value
    """
    win_home_team = y_true[:, 0:1]
    win_home_or_draw = y_true[:, 1:2]
    win_away = y_true[:, 2:3]
    win_away_or_draw = y_true[:, 3:4]
    draw = y_true[:, 4:5]
    no_bet = y_true[:, 5:6]
    odds_a = y_true[:, 6:7]
    odds_b = y_true[:, 7:8]
    gain_loss_vector = K.concatenate([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
      win_home_or_draw * (1/(1 -1/odds_b) - 1) + (1 - win_home_or_draw) * -1,
      win_away * (odds_b - 1) + (1 - win_away) * -1,
      win_away_or_draw * (1/(1 -1/odds_a) - 1) + (1 - win_away_or_draw) * -1,
      draw * (1/(1 - 1/odds_a - 1/odds_b) - 1) + (1 - draw) * -1,
      K.zeros_like(odds_a)], axis=1)
    return -1 * K.mean(K.sum(gain_loss_vector * y_pred, axis=1))