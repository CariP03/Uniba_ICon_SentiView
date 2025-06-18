:- dynamic token_at/2.

% Threshold
threshold(0.519).

% Base: if the word has a score, otherwise 0
token_base_score(Word, Score) :-
    word_score(Word, Score), !.
token_base_score(Word, Score) :-
    string_concat("not_", Base, Word),
    word_score(Base, BaseScore),
    Score is -BaseScore, !.
token_base_score(_, 0).

% Manage intensifiers
token_effective_score(Index, ScoreEff) :-
    token_at(Index, Word),
    token_base_score(Word, Base),
    % Intensifiers
    ( Index > 1,
      Prev is Index - 1,
      token_at(Prev, PrevW),
      intensifier(PrevW, M) ->
        ScoreEff is Base * M ;
        ScoreEff = Base ).

% Total Sum
sentiment_sum(Sum) :-
    findall(S, token_effective_score(_, S), Scores),
    sum_list(Scores, Sum).

% Non-zero tokens count 
sentiment_count_nonzero(Count) :-
    findall(1, (token_effective_score(_, S), S \= 0), L),
    length(L, Count).

% Total token count
sentiment_total_tokens(Tot) :-
    findall(1, token_at(_, _), L),
    length(L, Tot).

% Classifier
is_positive :-
    sentiment_sum(S),
    threshold(T),
    S >= T.
